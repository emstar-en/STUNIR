#include <stdint.h>
#include <stdbool.h>

// Define the Position struct for testing
typedef struct Position {
  uint8_t lat;
  uint8_t lon;
  uint8_t alt;
} Position;

// Test function with the generated pattern
struct Position get_position(uint8_t sysid) {
  uint8_t pos_lat = 0;
  uint8_t pos_lon = 0;
  uint8_t pos_alt = 0;
  return (struct Position){pos_lat, pos_lon, pos_alt};
}

// Test bool functions
bool arm_vehicle(uint8_t sysid, uint8_t compid) {
  bool success = false;
  /* nop */
  return success;
}

// Test int functions
int32_t parse_heartbeat(const uint8_t* buffer, uint8_t len) {
  int32_t msg_type = buffer[0];
  uint8_t result = 0;
  return result;
}

int main() {
  struct Position p = get_position(1);
  bool armed = arm_vehicle(1, 1);
  uint8_t buf[10] = {0};
  int32_t result = parse_heartbeat(buf, 10);
  return 0;
}
