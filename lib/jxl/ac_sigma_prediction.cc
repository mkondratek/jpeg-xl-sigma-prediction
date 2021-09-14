#include "ac_sigma_prediction.h"

namespace sigma_prediction {

const float* vertical_noise_evaluation = vertical_horizontal_noise_evaluation;
const float* horizontal_noise_evaluation = vertical_horizontal_noise_evaluation + 8;

void derive_sigmas(float dct1d[2 * 8], float sigmas[8][8]) {
  float mod = 0;
  for (int i = 0; i < 16; ++i) {
    mod += std::abs(dct1d[i]) * vertical_horizontal_noise_evaluation[i];
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      sigmas[i][j] = beta0[i][j] + mod * beta1[i][j];
    }
  }
}

}  // namespace sigma_prediction
