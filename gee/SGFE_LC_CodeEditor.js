/**
 * SGFE-LC — Code Editor companion
 * Cloud project: hallowed-chain-458005-g9
 *
 * 1. Replace SITES_ASSET after uploading the harmonized FeatureCollection.
 * 2. Run each EXPORT block as needed. Do not use product layers as predictors.
 */
var GEE_PROJECT = 'hallowed-chain-458005-g9';
var SITES_ASSET = 'projects/hallowed-chain-458005-g9/assets/sgfe_lc/ca_902_sites';
var MAP_YEAR = 2018;

var countries = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.inList('ADM0_NAME', [
    'Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Uzbekistan', 'Turkmenistan'
  ]));
var roi = countries.geometry().dissolve({maxError: 100});

var sites = ee.FeatureCollection(SITES_ASSET);

function prepL8(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
    .and(qa.bitwiseAnd(1 << 4).eq(0));
  var optical = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2);
  optical = optical.updateMask(mask);
  var nir = optical.select('SR_B5');
  var red = optical.select('SR_B4');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  return optical.addBands(ndvi).copyProperties(img, ['system:time_start']);
}

var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi).filterDate(MAP_YEAR + '-01-01', (MAP_YEAR + 1) + '-01-01')
  .map(prepL8);
var l8med = l8.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','NDVI']).median()
  .rename(['l8_B2','l8_B3','l8_B4','l8_B5','l8_B6','l8_B7','l8_NDVI']);

var aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(MAP_YEAR + '-01-01', (MAP_YEAR + 1) + '-01-01')
  .filterBounds(roi).mosaic();

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(roi).filterDate(MAP_YEAR + '-01-01', (MAP_YEAR + 1) + '-01-01');
var dwBands = ['water','trees','grass','flooded_vegetation','crops',
               'shrub_and_scrub','built','bare','snow_and_ice'];
var dwProb = dw.select(dwBands).mean();
var dwL1 = dwProb.toArray().arrayArgmax().arrayGet([0])
  .remap([0,1,2,3,4,5,6,7,8], [6,2,3,6,1,4,7,5,8], 0)
  .rename('dw_l1_2018');

var glcTile = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual')
  .filterBounds(roi).mosaic();
// b1=2000 ... b19=2018
var glc2018 = glcTile.select('b19').rename('glc_raw_2018');

var predictors = aef;  // independent observational / foundation features only
Map.centerObject(roi, 5);
Map.addLayer(roi, {color: '888888'}, 'Central Asia ROI', false);
Map.addLayer(aef.select(['A00','A01','A02']), {min: -1, max: 1}, 'AEF RGB', false);
Map.addLayer(dwL1.randomVisualizer(), {}, 'DW Level-1 2018', false);
Map.addLayer(sites, {color: 'ff0000'}, 'Field sites');

var samples = predictors.sampleRegions({
  collection: sites,
  properties: ['site_id','class_id','country','year','split','fold_id'],
  scale: 30,
  geometries: true,
  tileScale: 4
});

Export.table.toDrive({
  collection: samples,
  description: 'SGFE_sites_2018_AEF',
  folder: 'SGFE_LC',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: l8med.addBands(dwL1).addBands(glc2018).sampleRegions({
    collection: sites,
    properties: ['site_id','class_id','country','year'],
    scale: 30,
    geometries: true
  }),
  description: 'SGFE_sites_2018_L8_products',
  folder: 'SGFE_LC',
  fileFormat: 'CSV'
});
