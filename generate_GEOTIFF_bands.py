from osgeo import gdal, gdal_array, osr
import tifffile

# Load your data into a numpy array
filename = "results/interpolation_WI_order3.tif"
data = tifffile.imread(filename)

# Define the dimensions of your raster
if data.shape[0] == 6:
    bands, rows, cols = data.shape
else:
    rows, cols, bands = data.shape

# Define the output file name
output_file = "results/interpolation_WI_order3_GEO.tif"

# Create the output GeoTIFF file
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(output_file, cols, rows, bands, gdal.GDT_UInt16)  # Specify data type as UInt16

georef_data = "datasets/whole_images/RGBNED.tif"
georef_dataset = gdal.Open( georef_data, gdal.GA_ReadOnly )
projection = georef_dataset.GetProjection()
geotransform = georef_dataset.GetGeoTransform()

# Write your data into the bands of the output GeoTIFF file
for i in range(bands):
    output_dataset.GetRasterBand(i + 1).WriteArray(data[:, :, i])

# Set the geotransform 
output_dataset.SetGeoTransform(geotransform)

# Set the projection 
output_dataset.SetProjection(projection)

# Close the output dataset
output_dataset = None

print("GeoTIFF file saved successfully!")
