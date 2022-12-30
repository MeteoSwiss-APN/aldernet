&RunSpecification
 strict_nl_parsing  = .true.
 verbosity          = "moderate"
 diagnostic_length  = 110
 soft_memory_limit   = 29.0
 additional_diagnostic = .false.
 strict_usage = .false.
/

&GlobalResource
 dictionary           = "/oprusers/osm/opr/config/resources/dictionary_cosmo.txt",
 grib_definition_path = "/oprusers/osm/opr/config/resources/eccodes_definitions_cosmo",
                        "/oprusers/osm/opr/config/resources/eccodes_definitions_vendor"
 grib2_sample         = "/oprusers/osm/opr/config/resources/eccodes_samples/COSMO_GRIB2_default.tmpl"
 location_list        = "/oprusers/osm/opr/config/resources/location_list.txt"
/

&GlobalSettings
 default_model_name = "cosmo-1e"
 default_out_type_packing="simple,0"
 location_to_gridpoint="sn"
/

&ModelSpecification
 model_name         = "cosmo-1e"
 earth_axis_large   = 6371229.
 earth_axis_small   = 6371229.
/

#-------------------------------------------------------------------------------------------------------------
# In core data
#-------------------------------------------------------------------------------------------------------------
# (1) Define base grid
#-------------------------------------------------------------------------------------------------------------
&Process
  in_file = "/store/s83/osm/KENDA-1/ANA22/det/laf2022070600"
  out_type = "INCORE" /
&Process in_field = "HSURF", tag="HSURF" /
&Process in_field = "FR_LAND", tag="fr_land" /
&Process in_field = "FR_LAKE", tag="fr_lake" /
&Process in_field = "T_SO", levlist=0, tag="sst" /

#-------------------------------------------------------------------------------------------------------------
# Start of merge operations
#-------------------------------------------------------------------------------------------------------------
# ATTENTION: All merge operations have to be included in the &&Merge - &&EndMerge brackets
#-------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------
# End of merge operations
#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
# Products
#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
# Pollen assimilation
#-------------------------------------------------------------------------------------------------------------

&Process
  in_file="/store/s83/osm/KENDA-1/ANA22/det/laf<yyyymmddhh:2022020900>"
  out_file="22_cory_cosmo.atab"
  out_type="XLS_TABLE", out_type_text1="ATAB", out_type_noundef=.false.
  locgroup="pollen_obs"
  tstart=0, tstop=1224, tincr=1
/

&Process in_field="CORY", levlist=80/
