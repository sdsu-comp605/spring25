# 4) Introduction to Vectorization

Last time:

- Introduction to architectures
- Memory

Today:
1. Single-thread performance trends
2. Introduction to Vectorization

## 1. Single-thread performance trends
Single-thread performance has increased significantly
since ~2004 when clock frequency stagnated?

![42 years of microprocessor data](https://www.karlrupp.net/wp-content/uploads/2018/02/42-years-processor-trend.png)

This is a result of doing more per clock cycle.

![Flops per clock cycle](https://www.karlrupp.net/wp-content/uploads/2013/06/flops-per-cycle-sp.png)

Let's visit some slides:

* [Georg Hager (2019): Modern Computer Architucture](https://moodle.rrze.uni-erlangen.de/pluginfile.php/12916/mod_resource/content/6/01_IntroArchitecture.pdf)


### Further resources

* [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#)
* Wikichip
  * [Intel Xeon: Cascade Lake](https://en.wikichip.org/wiki/intel/microarchitectures/cascade_lake)
  * [AMD EPYC gen2: Rome](https://en.wikichip.org/wiki/amd/cores/rome)
  * [IBM POWER9](https://en.wikichip.org/wiki/ibm/microarchitectures/power9)
* [Agner Fog's website](https://www.agner.org/optimize/)

## 2. Introduction to Vectorization
