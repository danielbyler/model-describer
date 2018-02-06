import glob
import unittest


if __name__ == '__main__':

    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    test_files = glob.glob('*_tests.py')
    if len(test_files) == 0:
        test_files = glob.glob('./tests/*_tests.py')
    module_strings = [test_file[0:len(test_file) - 3].split('/')[-1] for test_file in test_files]
    print(module_strings)
    suites = [unittest.defaultTestLoader.loadTestsFromName(test_file) for test_file in module_strings]
    test_suite = unittest.TestSuite(suites)
    test_runner = unittest.TextTestRunner().run(test_suite)
