# rg-dataset

## **About**  
rg-dataset contains data for training and testing ML-based source finder applications. This dataset contains images (FITS format), regions (DS9 format) and masks (FITS format) for these classes of radio objects:

- Galaxies
- Sidelobes (or artefacts)
- Point-sources

**NB: These data include private observatory data that are to be kept private and not shared to the public unless explicitly agreed before.**

## **Usage**
This repository contains only the Data Version Control (DVC) files tracking the image data. Actual data are instead kept in a remote gDrive shared folder. To get access to the data and work on them do the following: 

1. Clone the repo:    

   ```git clone https://github.com/SKA-INAF/rg-dataset.git```
   
2. Download remote data:

   ```dvc pull```
   
3. Add or modify a file (e.g. img.fits) and commit the change:   

   ```dvc add img.fits```    (generate file .dvc)     
   ```git add .gitignore img.fits.dvc```       
   ```git commit -m ‘Added file’```    
   ```git push origin master```    
   ```dvc push```  (upload new/updated data in the remote storage)    

