/* STUNIR: C emission (raw target) */
/* module: web_example */
/* Web API example with REST endpoints */


#include <stdint.h>
#include <stdbool.h>

/* NOTE: This is a minimal, deterministic stub emitter.
 * It preserves IR ordering and emits placeholder bodies.
 */

/* fn: get_user */
const char* get_user(int32_t user_id) {
  /* TODO: implement */
  return NULL;
}

/* fn: create_user */
int32_t create_user(const char* name, const char* email) {
  /* TODO: implement */
  return 0;
}

/* fn: update_user */
bool update_user(int32_t user_id, const char* data) {
  /* TODO: implement */
  return false;
}

/* fn: delete_user */
bool delete_user(int32_t user_id) {
  /* TODO: implement */
  return false;
}

