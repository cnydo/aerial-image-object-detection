@echo off

rem Rename val folder to valid if needed
if exist val (
    ren valid val
)

rem Move images from train to train/images
mkdir train\images
for %%f in (train\*.jpg) do move "%%f" train\images

rem Move images from valid to valid/images
mkdir val\images
for %%f in (val\*.jpg) do move "%%f" val\images

rem Move images from test to test/images
mkdir test\images
for %%f in (test\*.jpg) do move "%%f" test\images