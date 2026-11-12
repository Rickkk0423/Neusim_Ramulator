#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {
struct Completion {
  uint64_t id;
  uint64_t cycle;
};

struct Engine {
  std::unique_ptr<Ramulator::IFrontEnd> frontend;
  std::unique_ptr<Ramulator::IMemorySystem> memory;
  std::deque<Completion> completions;
  uint64_t cycle = 0;
  bool finalized = false;
};
thread_local std::string last_error;

template <class F> int checked(F function) {
  try {
    function();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  } catch (...) {
    last_error = "Unknown exception in Ramulator bridge";
    return -1;
  }
}
}  // namespace

extern "C" {
const char* nr_error() { return last_error.c_str(); }

void* nr_create(const char* config_path) {
  Engine* result = nullptr;
  checked([&] {
    const auto config = Ramulator::Config::parse_config_file(config_path);
    auto engine = std::make_unique<Engine>();
    engine->frontend.reset(Ramulator::Factory::create_frontend(config));
    engine->memory.reset(Ramulator::Factory::create_memory_system(config));
    engine->frontend->connect_memory_system(engine->memory.get());
    engine->memory->connect_frontend(engine->frontend.get());
    if (engine->memory->get_tCK() <= 0 || engine->memory->get_tx_bytes() <= 0)
      throw std::runtime_error("DRAM returned invalid clock/transaction geometry");
    result = engine.release();
  });
  return result;
}

int nr_geometry(void* handle, double* tck_ns, int* transaction_bytes) {
  return checked([&] {
    auto& engine = *static_cast<Engine*>(handle);
    *tck_ns = engine.memory->get_tCK();
    *transaction_bytes = engine.memory->get_tx_bytes();
  });
}

// 1 = accepted, 0 = queue full, -1 = backend error. No exception crosses the ABI.
int nr_send(void* handle, uint64_t id, int type, int64_t address, int source) {
  bool accepted = false;
  int status = checked([&] {
    auto& engine = *static_cast<Engine*>(handle);
    const int tx = engine.memory->get_tx_bytes();
    if (engine.finalized || address < 0 || address % tx || (type != 0 && type != 1))
      throw std::runtime_error("Invalid or unaligned external request");
    accepted = engine.frontend->receive_external_requests(
        type, address, source,
        [&engine, id](Ramulator::Request&) {
          engine.completions.push_back({id, engine.cycle});
        }, tx);
  });
  return status < 0 ? status : static_cast<int>(accepted);
}

int nr_tick(void* handle, uint64_t count) {
  return checked([&] {
    auto& engine = *static_cast<Engine*>(handle);
    if (engine.finalized) throw std::runtime_error("Backend already finalized");
    for (uint64_t i = 0; i < count; ++i) {
      ++engine.cycle;
      engine.memory->tick();
    }
  });
}

int nr_pop(void* handle, uint64_t* id, uint64_t* cycle) {
  auto& engine = *static_cast<Engine*>(handle);
  if (engine.completions.empty()) return 0;
  const auto event = engine.completions.front();
  engine.completions.pop_front();
  *id = event.id;
  *cycle = event.cycle;
  return 1;
}

int nr_finalize(void* handle, const char* stats_path) {
  return checked([&] {
    auto& engine = *static_cast<Engine*>(handle);
    if (engine.finalized) return;
    engine.frontend->finalize();
    engine.memory->finalize();
    engine.frontend->update_stats_recursive();
    engine.memory->update_stats_recursive();
    std::ofstream output(stats_path);
    if (!output) throw std::runtime_error("Cannot open DRAM statistics output");
    engine.frontend->print_stats(output);
    engine.memory->print_stats(output);
    if (!output) throw std::runtime_error("Cannot write DRAM statistics");
    engine.finalized = true;
  });
}

void nr_destroy(void* handle) { delete static_cast<Engine*>(handle); }
}
