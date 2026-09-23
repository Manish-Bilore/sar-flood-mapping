#!/usr/bin/env bash
# netcheck.sh — real throughput to the actual data hosts (the Cloudflare test in sysprobe returned 23 B/s = blocked/failed, not a measurement)
set -u
t(){ printf '%-38s ' "$1"; curl -sL -o /dev/null --max-time 120 -w 'HTTP %{http_code}  %{size_download} B  %{speed_download} B/s  %{time_total}s\n' "$2"; }
# Copernicus GLO-30 COG tile over Kuttanad (N09 E076), public AWS bucket, ~20-40 MB
t "AWS COG (Cop-DEM, Kerala)" "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N09_00_E076_00_DEM/Copernicus_DSM_COG_10_N09_00_E076_00_DEM.tif"
# ASF search with valid params (the 400 in sysprobe was a missing-parameter error, not a block)
t "ASF search API (valid query)" "https://api.daac.asf.alaska.edu/services/search/param?platform=S1&processingLevel=SLC&intersectsWith=POINT(76.4%209.5)&start=2018-08-10&end=2018-08-25&output=json"
t "PC STAC search" "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc"
t "CDSE STAC" "https://stac.dataspace.copernicus.eu/v1/collections"
