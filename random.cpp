#include "random.h"
#include <stdlib.h>
//TODO:Implement
/*
float rnd_generate()
{
	float x = 1.0*rand()/RAND_MAX;
	if(x == 1) x = 0.999;
	if(x == 0) x = 0.001;
	return x;
}
*/

#include <stdlib.h>

typedef struct
{
	int     i24;
	int		j24;
	int		ncarry;
	int		ivec;
	float	seeds[25];

} random_num;


random_num ran_num[DIMENSIONS_COUNT];

bool initRandom(long XXX)
{
	for(int _ = 0; _ < DIMENSIONS_COUNT; _++) {
	long ijkl = XXX*DIMENSIONS_COUNT + _;

  long ij,kl,i,j,k,l,m;
  int ii,jj;
  float s,t;

  ij = ijkl/30082;
  kl = ijkl-30082*ij;
  i = (ij/177)%177 + 2;
  j = ij%177 +2;
  k = (kl/169)%178 + 1;
  l = kl%169;

  for(ii = 1; ii<=24; ii++)
	{
	 s = 0.0;
	 t = 0.5;
	 for(jj=1; jj<=24; jj++)
	  {
		m = ( ( (i*j)%179) * k ) % 179;
		i = j;
		j = k;
		k = m;
		l = (53*l+1) % 169;
		if ( (l*m)%64 >= 32 ) s = s+t;
		t = 0.5f*t;
	  }
	 ran_num[_].seeds[ii] = s;
	}
  ran_num[_].ncarry=0;
  
  ran_num[_].i24 = 24;
  ran_num[_].j24 = 10;
  ran_num[_].ivec = 1;
	}
    return true;
}


// returns lenv random numbers in array rvec
float getRandom(int _)
{
	float twop24,twom24,carry,uni;
	twop24=16777216.;
	twom24=1.f/twop24;
	if (ran_num[_].ncarry==4) carry=twom24;
	if (ran_num[_].ncarry==0) carry=0.;
	
	uni=ran_num[_].seeds[ran_num[_].i24]-ran_num[_].seeds[ran_num[_].j24]-carry;
	if (uni<0.)
	{
	uni=uni+1.f;
	carry=twom24;
	}
	else carry=0.;
	ran_num[_].seeds[ran_num[_].i24]=uni;
	ran_num[_].i24=ran_num[_].i24-1;
	if (ran_num[_].i24==0) ran_num[_].i24=24;
	ran_num[_].j24=ran_num[_].j24-1;
	if (ran_num[_].j24==0) ran_num[_].j24=24;

	if(uni == 0.0f)
	{
		uni = 0.0000000001f;
	}

	if(uni == 1.0f)
	{
		uni = 0.99999999999f;
	}

	return uni;

}
