import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

# -----------------------------
# Basic instrument parameters
# -----------------------------
dmd_pitch_um = 7.6  # replace with your DMD pitch
wavelength_um = 0.532  # green laser example

M_proj = 5.0  # DMD-to-SUT magnification
M_cam = 0.5  # SUT-to-camera magnification

NA_proj = 0.05  # projection lens NA
NA_cam = 0.05  # camera imaging lens NA

camera_pixel_um = 3.45  # camera pixel size
rows_on = 3  # DMD rows used to form the projected line

# Use a cropped DMD region first, not the whole chip
nx = 512
ny = 256

# -----------------------------
# Make DMD pattern
# -----------------------------
dmd = np.zeros((ny, nx), dtype=float)

row_center = ny // 2
dmd[row_center - rows_on // 2 : row_center + rows_on // 2 + 1, :] = 1.0

# Optional: Gaussian illumination envelope on DMD
x = (np.arange(nx) - nx / 2) * dmd_pitch_um
y = (np.arange(ny) - ny / 2) * dmd_pitch_um
X, Y = np.meshgrid(x, y)

beam_radius_um = 0.45 * nx * dmd_pitch_um
illum = np.exp(-(X**2 + Y**2) / beam_radius_um**2)

dmd_illuminated = dmd * illum

# -----------------------------
# Project DMD onto SUT
# -----------------------------
# DMD pixel pitch mapped to SUT
sut_pixel_um = dmd_pitch_um * M_proj

# Approximate diffraction-limited projection blur on SUT
# This is a crude Gaussian approximation, not a full Airy model.
proj_blur_um = wavelength_um / (2 * NA_proj)
proj_blur_px = proj_blur_um / sut_pixel_um

sut_irradiance = gaussian_filter(dmd_illuminated, sigma=proj_blur_px)

# -----------------------------
# Perfect diffuse SUT
# -----------------------------
# For a perfect diffuse flat SUT, camera sees projected irradiance
sut_reflectance = 1.0
sut_radiance = sut_reflectance * sut_irradiance

# -----------------------------
# Image SUT onto camera
# -----------------------------
# SUT sampling projected onto camera sensor
sensor_sample_um = sut_pixel_um * M_cam

# Approximate imaging blur on sensor
cam_blur_um = wavelength_um / (2 * NA_cam)
cam_blur_px = cam_blur_um / sensor_sample_um

sensor_image = gaussian_filter(sut_radiance, sigma=cam_blur_px)

# Resample to actual camera pixels
resample_factor = sensor_sample_um / camera_pixel_um
camera_image = zoom(sensor_image, zoom=resample_factor, order=1)

# -----------------------------
# Plot results
# -----------------------------
plt.figure()
plt.imshow(dmd_illuminated, cmap="gray")
plt.title("DMD pattern")
plt.xlabel("DMD pixel")
plt.ylabel("DMD pixel")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(sut_irradiance, cmap="gray")
plt.title("Projected irradiance on perfect diffuse SUT")
plt.xlabel("SUT sample")
plt.ylabel("SUT sample")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(camera_image, cmap="gray")
plt.title("Simulated camera image")
plt.xlabel("Camera pixel")
plt.ylabel("Camera pixel")
plt.colorbar()
plt.show()

# Plot cross-section through the line
mid = camera_image.shape[0] // 2
plt.figure()
plt.plot(camera_image[:, camera_image.shape[1] // 2])
plt.title("Camera vertical line profile")
plt.xlabel("Camera pixel")
plt.ylabel("Intensity")
plt.show()
