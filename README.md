# SwitchbackFinder

A collection of functions for downloading and cleaning Parker Solar Probe and Solar Orbiter data. After cleaning the data an automated algorithm is used to identify magnetic switchbacks

  - Organize the magnetic field data by month for analysis.
  - Identify any gaps in the magnetic field time series data.
  - To provide a meaningful comparison between intervals at different distances, resample the time series at a constant sampling time of 1 second.
  - Utilize the resampled time series to identify events of interest.
  - Remove any events corresponding to times related to the identified gaps in the data, with a threshold of 10 seconds for gap duration.
  - Plot the low resolution magnetic field data by month and identify any Heliospheric Current Sheet (HCS) crossings. Remove any events corresponding to HCS crossings or other transients.
  - Consider the duration of the identified events and convert the temporal scales to spatial scales using the modified Taylor's hypothesis, $\ell =V_{tot}* \delta \tau$, where $V_{tot} = |\boldsymbol{V_{sw} + \boldsymbol{V}{a} - \boldsymbol{V}{sc}}|$, and $d_i$ is the ion inertial length.
  - Due to poor data quality, data from the Parker Solar Probe (PSP) beyond 0.5 astronomical units (AU) should not be used. Instead, data from the Solar Orbiter (SoLO) should be utilized for analysis between 0.5 and 1 AU.

# Installation  (Please take a look at the environemnt found in my repository: MHDTurbPy)
  - create conda enviroment
  
```bash
conda env create -n MHDTurbPy --file /path/to/environment.yml 
```

 - Download the package
``` bash
git clone https://github.com/nsioulas/SwitchbackFinder
/
```

# Usage

Some examples of how to download, visualize data can be found in ```SB_identification_public.ipynb```

# Contact
If you have any questions, please don't hesitate to reach out to nsioulas@g.ucla.edu.

# Citation

If you use this work, please cite:

```
@ARTICLE{2021ApJ...919L..31T,
       author = {{Tenerani}, Anna and {Sioulas}, Nikos and {Matteini}, Lorenzo and {Panasenco}, Olga and {Shi}, Chen and {Velli}, Marco},
        title = "{Evolution of Switchbacks in the Inner Heliosphere}",
      journal = {\apjl},
     keywords = {Heliosphere, Solar wind, Interplanetary turbulence, Alfven waves, Space plasmas, 711, 1534, 830, 23, 1544, Astrophysics - Solar and Stellar Astrophysics, Physics - Plasma Physics, Physics - Space Physics},
         year = 2021,
        month = oct,
       volume = {919},
       number = {2},
          eid = {L31},
        pages = {L31},
          doi = {10.3847/2041-8213/ac260610.48550/arXiv.2109.06341},
archivePrefix = {arXiv},
       eprint = {2109.06341},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJ...919L..31T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}



```



