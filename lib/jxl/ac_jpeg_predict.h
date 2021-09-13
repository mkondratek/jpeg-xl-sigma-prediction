#ifndef JPEGXL_AC_JPEG_PREDICT_H
#define JPEGXL_AC_JPEG_PREDICT_H

namespace individual_project {
const float beta0[8][8] = {{0.12161, 0.0431451, 0.0167912, 0.00981121,
                            0.00641885, 0.00539675, 0.00408265, 0.00391835},
                           {0.0446049, 0.0161903, 0.00764332, 0.00464828,
                            0.00334202, 0.0027926, 0.0028659, 0.0025634},
                           {0.015422, 0.00757373, 0.00518219, 0.0040314,
                            0.00310594, 0.00276036, 0.00279747, 0.00262059},
                           {0.008634, 0.00408438, 0.00310416, 0.00283518,
                            0.00249884, 0.00261862, 0.00270079, 0.00272077},
                           {0.00517393, 0.00219898, 0.00230598, 0.00229765,
                            0.00249455, 0.00264824, 0.00281668, 0.00268258},
                           {0.00483147, 0.00213903, 0.00203529, 0.00233784,
                            0.00266754, 0.00279189, 0.00312634, 0.00304787},
                           {0.00422391, 0.00230999, 0.00233345, 0.00245605,
                            0.00268812, 0.00300985, 0.00345132, 0.00357965},
                           {0.00418222, 0.00226478, 0.0022221, 0.00229315,
                            0.00243802, 0.00284204, 0.00333675, 0.00429227}};

const float beta1[8][8] = {{0.820167, 0.173436, 0.065358, 0.0285933, 0.030308,
                            0.0263405, 0.0245402, 0.025416},
                           {0.507648, 0.161671, 0.104586, 0.0640292, 0.0480852,
                            0.0370606, 0.0307728, 0.029797},
                           {0.464268, 0.167816, 0.112236, 0.0723446, 0.05276,
                            0.0409698, 0.0322152, 0.0312013},
                           {0.353906, 0.151591, 0.112808, 0.0759643, 0.0563094,
                            0.043081, 0.0347454, 0.0336971},
                           {0.271047, 0.125342, 0.092469, 0.06893, 0.0507984,
                            0.0388497, 0.0327098, 0.0351324},
                           {0.187899, 0.0951923, 0.0743436, 0.0557639,
                            0.0433164, 0.0356416, 0.0304976, 0.0325903},
                           {0.143047, 0.0749104, 0.060284, 0.0474609, 0.0383998,
                            0.0330628, 0.0298227, 0.0303295},
                           {0.12602, 0.0678936, 0.05384, 0.0453382, 0.0385935,
                            0.0341108, 0.0282731, 0.0356609}};

const float vertical_horizontal_noise_evaluation[16] = {
    0., 0.0721139, 0.134336, 0.186764, 0.258038, 0.314061, 0.384939, 0.374337,
    0., 0.0721139, 0.134336, 0.186764, 0.258038, 0.314061, 0.384939, 0.374337
};

const float* vertical_noise_evaluation = vertical_horizontal_noise_evaluation;
const float* horizontal_noise_evaluation = vertical_horizontal_noise_evaluation + 8;

}  // namespace individual_project

#endif  // JPEGXL_AC_JPEG_PREDICT_H
