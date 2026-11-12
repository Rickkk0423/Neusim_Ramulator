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

namespace Ramulator {
// The upstream External frontend reports one source. Controllers size their
// per-source statistics from get_num_cores(), so multi-core traffic needs this
// small frontend adapter rather than passing out-of-range source IDs.
class NeuSimExternal : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, NeuSimExternal, "NeuSimExternal")

  int cores = 1;
  uint64_t capacity = 0;

 public:
  void init() override {
    m_clock_ratio = 1;
    cores = ParamReader<int>{m_config, "num_cores", m_name}.required();
    if (cores < 1) throw std::runtime_error("num_cores must be positive");
    const auto bytes = ParamReader<int64_t>{m_config, "address_capacity_bytes", m_name}.required();
    if (bytes <= 0) throw std::runtime_error("address_capacity_bytes must be positive");
    capacity = static_cast<uint64_t>(bytes);
  }
  void tick() override {}
  bool is_finished() override { return false; }
  int get_num_cores() override { return cores; }
  uint64_t get_capacity_bytes() const { return capacity; }
  bool receive_external_requests(int type, Addr_t address, int source,
                                 std::function<void(Request&)> callback,
                                 int bytes) override {
    if (source < 0 || source >= cores)
      throw std::runtime_error("Source ID exceeds the configured frontend core count");
    if (bytes <= 0 || static_cast<uint64_t>(bytes) > capacity || address < 0 ||
        static_cast<uint64_t>(address) > capacity - static_cast<uint64_t>(bytes))
      throw std::runtime_error("Request exceeds configured DRAM address capacity");
    Request request(address, type, source, std::move(callback));
    request.size_bytes = bytes;
    return m_memory_system->send(request);
  }
};
}  // namespace Ramulator

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
  uint64_t capacity = 0;
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
int nr_abi_version() { return 2; }
const char* nr_error() { return last_error.c_str(); }

void* nr_create(const char* config_path) {
  Engine* result = nullptr;
  checked([&] {
    const auto config = Ramulator::Config::parse_config_file(config_path);
    auto engine = std::make_unique<Engine>();
    engine->frontend.reset(Ramulator::Factory::create_frontend(config));
    auto* frontend = dynamic_cast<Ramulator::NeuSimExternal*>(engine->frontend.get());
    if (!frontend) throw std::runtime_error("Bridge requires the NeuSimExternal frontend");
    engine->capacity = frontend->get_capacity_bytes();
    engine->memory.reset(Ramulator::Factory::create_memory_system(config));
    engine->frontend->connect_memory_system(engine->memory.get());
    engine->memory->connect_frontend(engine->frontend.get());
    if (engine->memory->get_tCK() <= 0 || engine->memory->get_tx_bytes() <= 0)
      throw std::runtime_error("DRAM returned invalid clock/transaction geometry");
    result = engine.release();
  });
  return result;
}

int nr_geometry(void* handle, double* tck_ns, int* transaction_bytes, uint64_t* capacity_bytes) {
  return checked([&] {
    auto& engine = *static_cast<Engine*>(handle);
    *tck_ns = engine.memory->get_tCK();
    *transaction_bytes = engine.memory->get_tx_bytes();
    *capacity_bytes = engine.capacity;
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
