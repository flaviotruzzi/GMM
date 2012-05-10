#include <math.h>


static inline float
fastpow2 (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  float z = clipp - w + offset;
  union { int i; float f; } v = { (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) };

  return v.f;
}

float fast_exp_mineiro (float p)
{
  return fastpow2 (1.442695040f * p);
}




// Cawley's modification of Schraudolph's hack for fast exp() approximations
// NeCo 2 (9). pp. 2009-2012

#define EXP_A (1048576/M_LN2)
#define EXP_C 60801

inline double fast_exp_schraudolph(double y)
{
  union
  {
    double d;
#ifdef BIG_ENDIAN
    struct { int i, j; } n;
#else 
    struct { int j, i; } n;
#endif
  }
  _eco;
  _eco.n.i = (int)(EXP_A*(y)) + (1072693248 - EXP_C);
  _eco.n.j = 0;
  return _eco.d;
}
