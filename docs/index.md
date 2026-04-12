# Continuous Intelligence

This site provides documentation for this project.
Use the navigation to explore module-specific materials.

## Custom Project

### Dataset
I used "NCDC Hourly Global Surface Variables-Selected Subset"
to gather times and temperature data from "Sacramento Airport Surface Temperatures"
I filtered the temperatures down so they only had a TMP_Q_CODE of 5 which corresponded to reasonable.
Then temperature units are a bit confusing as apparently they are in celsius and multiploed by a faxto of 10.

### Signals
I utilized the temperature data to create rolling "curent" tempeature means and standard deviations. This was compared to a trailing baseline.

### Experiments
I attempted a few thermal datasets from Kaggle initially, but drift was impossible to detect as those datasets were apparently ill sampled. The final dataset used took one measurement per hour so I tried adjusting the trailing baseline window and currenly sized window.

### Results
The results are difficult to interpret. I overlooked a major scaling detail,
the threshold visual artifact:
C:\Repos\cintel-05-drift-detection\artifacts\threshold_colored_alex.png
Does present a nice display of signals verus thresholds.

### Interpretation
This data was du=ifficult to work with so I can not draw any business nor analytical insights. One thing that stood out to me was the consistency of the surface ground air temperature at the airport. It is most likelt caused by frequent high heat transport.

## How-To Guide

Many instructions are common to all our projects.

See
[⭐ **Workflow: Apply Example**](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
to get these projects running on your machine.

## Project Documentation Pages (docs/)

- **Home** - this documentation landing page
- **Project Instructions** - instructions specific to this module
- **Your Files** - how to copy the example and create your version
- **Glossary** - project terms and concepts

## Additional Resources

- [Suggested Datasets](https://denisecase.github.io/pro-analytics-02/reference/datasets/cintel/)
