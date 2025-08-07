# Empirical methods to determine surface air temperature from satellite-retrieved data

## Objectives
Surface air temperature (SAT) is an essential climate variable (ECV), which is frequently used for weather monitoring and prediction. However, it cannot be directly retrieved from satellite observations. Thus, in this study several methods are compared to identify the most accurate method and assess the validity of previously published empirical equations over a broader time period and spatial extent.

## Project structure

This project was part of my first published article and my Master's thesis. It was initially developed rapidly, due to deadline time, for use in a local environment. However, I am currently restructuring and improving the code to ensure better organization, OS compatibility, and usability. A public version will be uploaded so that anyone can download, run, and adapt the scripts for their own research or applications. 

Data section is not upload because AEMET's data is not public. For more information do not hesitate to contact me.

**Current project structure**

```
.
├── scripts/
│   ├── Database_creation_scripts 
│   │   ├── Albedo_1km_files.py # open albedo files (.nc) and reshape them to 1km resolution to get the data for the AEMET stations
│   │   ├── ERA5_datafile.py # open ERA5 files (grib) and get the data for the AEMET stations
│   │   ├── LST_stations.py # open LST files (.nc) and get the data for the AEMET stations
│   │   ├── MODIS_data_filters.py # script which filters and creates daily final MODIS data
│   │   ├── MODIS_datafusion.py #  script which merges all MODIS data deleting those days when some variable is missing
│   │   ├── NDVI_stations.py # open NDVI files (.nc) and get the data for the AEMET stations
│   │   ├── crete_geofile.py # extraction of geographical data of AEMET stations from DEM. Slope, aspect calculated on QGIS. Mean altitud calculaded on SNAP. Other data are extracted from AEMET files
│   │   ├── functions_SAT_inst.py
│   │   ├── interp_finalDB.py # script for AEMET and ERA5 data temporal interpolation
│   │   └── solar_coord.py # calculates solar coordinates with equations from Valor et al. (2023)
│   ├── Linear_equations_comparison
│   │   └── SAT_Niclos2014_eq.py # script to retrive SAT from Niclos et al. (2014) equations which were obtained for the summer period over the Valencian authonomy using a stepwise method
│   ├── Linear_methods_scripts
│   │   ├── SAT_linear_methods.py # functions script with linear methods applied (OLS, Ridge, Lasso and EN)
│   │   ├── linear_methods_func.py # main script for linear part which call functions and calculate statistics
│   │   └── select_linear_var.py # function which let user select variables that enter in models
│   └── Nonlinear_methods_scripts
│       ├── SAT_nonlinear_methods.py # functions script with nonlinear methods applied (RF,XGB, KNN and MLP)
│       ├── nonlinear_methods_func.py # main script for nonlinear part which call functions and calculate statistics
│       └── select_nonlinear_var.py # function which let user select variables that enter in models
├── LICENSE
└── README.md
```



## Project Status
This is a finished project, but the current version is being cleaned and prepared for public release. Paths, modularity, and documentation are being improved for OS-independence and reproducibility.

## Publication
*Joan Vedrí, Raquel Niclòs, Lluís Pérez-Planells, Enric Valor, Yolanda Luna, María José Estrel (2025)*. **Empirical methods to determine surface air temperature from satellite-retrieved data**. *International Journal of Applied Earth Observation and Geoinformation*, 135. 

[DOI: 10.1016/j.jag.2025.104380](https://doi.org/10.1016/j.jag.2025.104380)
