#ifndef CORRELATE_H
#define CORRELATE_H

void correlate(int ny, int nx, const float* data, float* result);
void correlate_sequential(int ny, int nx, const float* data, float* result);

#endif
