import numpy
import tensorflow as tf

def sp0Filters():
    ''' Steerable pyramid filters.  Transform described  in:

        @INPROCEEDINGS{Simoncelli95b,
            TITLE = "The Steerable Pyramid: A Flexible Architecture for
                     Multi-Scale Derivative Computation",
            AUTHOR = "E P Simoncelli and W T Freeman",
            BOOKTITLE = "Second Int'l Conf on Image Processing",
            ADDRESS = "Washington, DC", MONTH = "October", YEAR = 1995 }

        Filter kernel design described in:

        @INPROCEEDINGS{Karasaridis96,
            TITLE = "A Filter Design Technique for 
                     Steerable Pyramid Image Transforms",
            AUTHOR = "A Karasaridis and E P Simoncelli",
            BOOKTITLE = "ICASSP",	ADDRESS = "Atlanta, GA",
            MONTH = "May",	YEAR = 1996 }  '''
    filters = {}
    filters['harmonics'] = tf.constant([0], dtype=tf.float64)
    filters['lo0filt'] =  ( 
        tf.constant([[-4.514000e-04, -1.137100e-04, -3.725800e-04, -3.743860e-03, 
                   -3.725800e-04, -1.137100e-04, -4.514000e-04], 
                  [-1.137100e-04, -6.119520e-03, -1.344160e-02, -7.563200e-03, 
                    -1.344160e-02, -6.119520e-03, -1.137100e-04],
                  [-3.725800e-04, -1.344160e-02, 6.441488e-02, 1.524935e-01, 
                    6.441488e-02, -1.344160e-02, -3.725800e-04], 
                  [-3.743860e-03, -7.563200e-03, 1.524935e-01, 3.153017e-01, 
                    1.524935e-01, -7.563200e-03, -3.743860e-03], 
                  [-3.725800e-04, -1.344160e-02, 6.441488e-02, 1.524935e-01, 
                    6.441488e-02, -1.344160e-02, -3.725800e-04],
                  [-1.137100e-04, -6.119520e-03, -1.344160e-02, -7.563200e-03, 
                    -1.344160e-02, -6.119520e-03, -1.137100e-04], 
                  [-4.514000e-04, -1.137100e-04, -3.725800e-04, -3.743860e-03,
                    -3.725800e-04, -1.137100e-04, -4.514000e-04]], dtype=tf.float64) )
    filters['lofilt'] = (
        tf.constant([[-2.257000e-04, -8.064400e-04, -5.686000e-05, 8.741400e-04, 
                   -1.862800e-04, -1.031640e-03, -1.871920e-03, -1.031640e-03,
                   -1.862800e-04, 8.741400e-04, -5.686000e-05, -8.064400e-04,
                   -2.257000e-04],
                  [-8.064400e-04, 1.417620e-03, -1.903800e-04, -2.449060e-03, 
                    -4.596420e-03, -7.006740e-03, -6.948900e-03, -7.006740e-03,
                    -4.596420e-03, -2.449060e-03, -1.903800e-04, 1.417620e-03,
                    -8.064400e-04],
                  [-5.686000e-05, -1.903800e-04, -3.059760e-03, -6.401000e-03,
                    -6.720800e-03, -5.236180e-03, -3.781600e-03, -5.236180e-03,
                    -6.720800e-03, -6.401000e-03, -3.059760e-03, -1.903800e-04,
                    -5.686000e-05],
                  [8.741400e-04, -2.449060e-03, -6.401000e-03, -5.260020e-03, 
                   3.938620e-03, 1.722078e-02, 2.449600e-02, 1.722078e-02, 
                   3.938620e-03, -5.260020e-03, -6.401000e-03, -2.449060e-03, 
                   8.741400e-04], 
                  [-1.862800e-04, -4.596420e-03, -6.720800e-03, 3.938620e-03,
                    3.220744e-02, 6.306262e-02, 7.624674e-02, 6.306262e-02,
                    3.220744e-02, 3.938620e-03, -6.720800e-03, -4.596420e-03,
                    -1.862800e-04],
                  [-1.031640e-03, -7.006740e-03, -5.236180e-03, 1.722078e-02, 
                    6.306262e-02, 1.116388e-01, 1.348999e-01, 1.116388e-01, 
                    6.306262e-02, 1.722078e-02, -5.236180e-03, -7.006740e-03,
                    -1.031640e-03],
                  [-1.871920e-03, -6.948900e-03, -3.781600e-03, 2.449600e-02,
                    7.624674e-02, 1.348999e-01, 1.576508e-01, 1.348999e-01,
                    7.624674e-02, 2.449600e-02, -3.781600e-03, -6.948900e-03,
                    -1.871920e-03],
                  [-1.031640e-03, -7.006740e-03, -5.236180e-03, 1.722078e-02,
                    6.306262e-02, 1.116388e-01, 1.348999e-01, 1.116388e-01,
                    6.306262e-02, 1.722078e-02, -5.236180e-03, -7.006740e-03,
                    -1.031640e-03], 
                  [-1.862800e-04, -4.596420e-03, -6.720800e-03, 3.938620e-03,
                    3.220744e-02, 6.306262e-02, 7.624674e-02, 6.306262e-02,
                    3.220744e-02, 3.938620e-03, -6.720800e-03, -4.596420e-03,
                    -1.862800e-04],
                  [8.741400e-04, -2.449060e-03, -6.401000e-03, -5.260020e-03,
                   3.938620e-03, 1.722078e-02, 2.449600e-02, 1.722078e-02, 
                   3.938620e-03, -5.260020e-03, -6.401000e-03, -2.449060e-03,
                   8.741400e-04],
                  [-5.686000e-05, -1.903800e-04, -3.059760e-03, -6.401000e-03,
                    -6.720800e-03, -5.236180e-03, -3.781600e-03, -5.236180e-03,
                    -6.720800e-03, -6.401000e-03, -3.059760e-03, -1.903800e-04,
                    -5.686000e-05],
                  [-8.064400e-04, 1.417620e-03, -1.903800e-04, -2.449060e-03,
                    -4.596420e-03, -7.006740e-03, -6.948900e-03, -7.006740e-03,
                    -4.596420e-03, -2.449060e-03, -1.903800e-04, 1.417620e-03,
                    -8.064400e-04], 
                  [-2.257000e-04, -8.064400e-04, -5.686000e-05, 8.741400e-04,
                   -1.862800e-04, -1.031640e-03, -1.871920e-03, -1.031640e-03,
                    -1.862800e-04, 8.741400e-04, -5.686000e-05, -8.064400e-04,
                    -2.257000e-04]], dtype=tf.float64) )
    filters['mtx'] = tf.constant([ 1.000000 ],dtype=tf.float64)
    filters['hi0filt'] = ( 
        tf.constant([[5.997200e-04, -6.068000e-05, -3.324900e-04, -3.325600e-04, 
                   -2.406600e-04, -3.325600e-04, -3.324900e-04, -6.068000e-05, 
                   5.997200e-04],
                  [-6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04, 
                    -3.732100e-04, 1.459700e-04, 4.927100e-04, 1.263100e-04, 
                    -6.068000e-05],
                  [-3.324900e-04, 4.927100e-04, -1.616650e-03, -1.437358e-02, 
                    -2.420138e-02, -1.437358e-02, -1.616650e-03, 4.927100e-04, 
                    -3.324900e-04], 
                  [-3.325600e-04, 1.459700e-04, -1.437358e-02, -6.300923e-02, 
                    -9.623594e-02, -6.300923e-02, -1.437358e-02, 1.459700e-04, 
                    -3.325600e-04],
                  [-2.406600e-04, -3.732100e-04, -2.420138e-02, -9.623594e-02, 
                    8.554893e-01, -9.623594e-02, -2.420138e-02, -3.732100e-04, 
                    -2.406600e-04],
                  [-3.325600e-04, 1.459700e-04, -1.437358e-02, -6.300923e-02, 
                    -9.623594e-02, -6.300923e-02, -1.437358e-02, 1.459700e-04, 
                    -3.325600e-04], 
                  [-3.324900e-04, 4.927100e-04, -1.616650e-03, -1.437358e-02, 
                    -2.420138e-02, -1.437358e-02, -1.616650e-03, 4.927100e-04, 
                    -3.324900e-04], 
                  [-6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04, 
                    -3.732100e-04, 1.459700e-04, 4.927100e-04, 1.263100e-04, 
                    -6.068000e-05], 
                  [5.997200e-04, -6.068000e-05, -3.324900e-04, -3.325600e-04, 
                   -2.406600e-04, -3.325600e-04, -3.324900e-04, -6.068000e-05, 
                   5.997200e-04]]) )
    filters['bfilts'] = ( 
        tf.constant([-9.066000e-05, -1.738640e-03, -4.942500e-03, -7.889390e-03, 
                   -1.009473e-02, -7.889390e-03, -4.942500e-03, -1.738640e-03, 
                   -9.066000e-05, -1.738640e-03, -4.625150e-03, -7.272540e-03, 
                   -7.623410e-03, -9.091950e-03, -7.623410e-03, -7.272540e-03, 
                   -4.625150e-03, -1.738640e-03, -4.942500e-03, -7.272540e-03, 
                   -2.129540e-02, -2.435662e-02, -3.487008e-02, -2.435662e-02, 
                   -2.129540e-02, -7.272540e-03, -4.942500e-03, -7.889390e-03, 
                   -7.623410e-03, -2.435662e-02, -1.730466e-02, -3.158605e-02, 
                   -1.730466e-02, -2.435662e-02, -7.623410e-03, -7.889390e-03,
                   -1.009473e-02, -9.091950e-03, -3.487008e-02, -3.158605e-02, 
                   9.464195e-01, -3.158605e-02, -3.487008e-02, -9.091950e-03, 
                   -1.009473e-02, -7.889390e-03, -7.623410e-03, -2.435662e-02, 
                   -1.730466e-02, -3.158605e-02, -1.730466e-02, -2.435662e-02, 
                   -7.623410e-03, -7.889390e-03, -4.942500e-03, -7.272540e-03, 
                   -2.129540e-02, -2.435662e-02, -3.487008e-02, -2.435662e-02, 
                   -2.129540e-02, -7.272540e-03, -4.942500e-03, -1.738640e-03, 
                   -4.625150e-03, -7.272540e-03, -7.623410e-03, -9.091950e-03, 
                   -7.623410e-03, -7.272540e-03, -4.625150e-03, -1.738640e-03,
                   -9.066000e-05, -1.738640e-03, -4.942500e-03, -7.889390e-03,
                   -1.009473e-02, -7.889390e-03, -4.942500e-03, -1.738640e-03,
                   -9.066000e-05], dtype=tf.float64) )
    filters['bfilts'] = tf.reshape(filters['bfilts'],[filters['bfilts'].get_shape().as_list()[0],1])

    return filters
