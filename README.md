# Unscented Kalman Filter

This package implements the square root version of the nonlinear Unscented Kalman Filter in python. It uses a version of the cholesky update function similar to that found in MATLAB.

# Installation

Install using pip:
`pip install pyUKFsqrt`

Or install from source:
`python ./setup.py install`

# References

For the original references about the UKF and sqrt UKF, see:

@inproceedings{10.1117/12.280797,
    author = {Simon J. Julier and Jeffrey K. Uhlmann},
    title = {{New extension of the Kalman filter to nonlinear systems}},
    volume = {3068},
    booktitle = {Signal Processing, Sensor Fusion, and Target Recognition VI},
    editor = {Ivan Kadar},
    organization = {International Society for Optics and Photonics},
    publisher = {SPIE},
    pages = {182 -- 193},
    year = {1997},
    doi = {10.1117/12.280797},
    URL = {https://doi.org/10.1117/12.280797}
}

@INPROCEEDINGS{882463,
  author={E. A. {Wan} and R. {Van Der Merwe}},
  booktitle={Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No.00EX373)}, 
  title={The unscented Kalman filter for nonlinear estimation}, 
  year={2000},
  volume={},
  number={},
  pages={153-158},
  doi={10.1109/ASSPCC.2000.882463}
}

@inproceedings{VanderMerwe,
  doi = {10.1109/icassp.2001.940586},
  url = {https://doi.org/10.1109/icassp.2001.940586},
  publisher = {{IEEE}},
  author = {R. Van der Merwe and E.A. Wan},
  title = {The square-root unscented Kalman filter for state and parameter-estimation},
  booktitle = {2001 {IEEE} International Conference on Acoustics,  Speech,  and Signal Processing. Proceedings (Cat. No.01CH37221)}
}

# What is "Unscented"?

It means nothing. From the inventor, Jeffrey Uhlmann:

> "Initially I only referred to it as the “new filter.” Needing a more specific name, people in my lab began referring to it as the “Uhlmann filter,” which obviously isn’t a name that I could use, so I had to come up with an official term. One evening everyone else in the lab was at the Royal Opera House, and as I was working I noticed someone’s deodorant on a desk. The word “unscented” caught my eye as the perfect technical term. At first people in the lab thought it was absurd—which is okay because absurdity is my guiding principle—and that it wouldn’t catch on. My claim was that people simply accept technical terms as technical terms: for example, does anyone think about why a tree is called a tree? Within a few months we had a speaker visit from another university who talked about his work with the “unscented filter.” Clearly he had never thought about the origin of the term. The cover of the issue of the March 2004 Proceedings we’re discussing right now has “Unscented” in large letters on the cover, which shows that it has been accepted as the technical term for that approach." [First-Hand:The_Unscented_Transform](https://ethw.org/First-Hand:The_Unscented_Transform)