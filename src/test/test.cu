#include <stdio.h>
#include "../fake_quantize.h"

int main(int argc, char *argv[])
{
	Tensor input = randn({2, 2});
	fake_quantize(input, 8);
	return 0;
}
