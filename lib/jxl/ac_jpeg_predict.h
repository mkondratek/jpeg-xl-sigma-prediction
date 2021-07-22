#ifndef JPEGXL_AC_JPEG_PREDICT_H
#define JPEGXL_AC_JPEG_PREDICT_H

namespace individual_project {

template<typename T>
void predict(T* ac, const T* top_ac, const T* left_ac, int c,
             bool inplace, bool is_transposed);

void applyPrediction(int32_t* ac, const int32_t* predictions, size_t row_size);
}  // namespace individual_project

#endif  // JPEGXL_AC_JPEG_PREDICT_H
