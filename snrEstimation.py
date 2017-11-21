import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy


def usage():
  print("""Usage:
           -o, --output [output_file_name] Output file name (Required)
           -h, --help                      Print this message (Optional)

        """)

def init_params():
  '''
  Initializes the parameters for the log and plot directory.
  '''
  log_file_directory = 'logs'
  plots_directory = 'plots'

def process_args():
  '''
  Reads the arguments from the command line. The acceptable arguments are
  -h and -o. Processes them as required.
  '''
  print("Processing arguments ...")
  try:
    opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help", "output="])
  except getopt.GetoptError as err:
    print(str(err))
    usage()
    sys.exit(2)

  global output_file_name, scenario_name
  for opt, arg in opts:
      if opt in ("-h", "--help"):
          usage()
          sys.exit()
      elif opt in ("-o", "--output"):
          output_file_name = arg
  try:
      output_file_name
  except:
      print("Error: Output file name not defined")
      usage()
      sys.exit(2)
  else:
      # output_file_name = log_file_directory + "/" + output_file_name
      output_file_name_split = output_file_name.split('_')
      scenario_name = ""
      for k,n in enumerate(output_file_name_split[1:]):
          scenario_name += n
          if k != len(output_file_name_split[1:]) -1:
              scenario_name += '_'

  print("---Output File Name: " + output_file_name)

  print("Done processing argument!")
  print("")

def get_complex_output():
  """
  Reads from a file at output_file_name as a complex 64 bit array. Then reshapes
  this into a column vector and returns.
  """
  print("Getting output waveform...")
  complex_output= np.fromfile(output_file_name, dtype = 'complex64').reshape(-1,1)   
  print("---Length: " + str(len(complex_output)))
  print("Done getting output waveform!")
  print("")
  return complex_output

def process_output(complex_output):
  '''
  Reads the complex output vector. Using a rolling window of window_length,
  tries to figure out the windowing for when we have silence (and ambient noise)
  and when there is signal received. Based on this, estimates the SNR for the
  given timestep, and produces an array of tuples, each of the type,
  (timestep, SNR)
  '''
  t = np.arange(0, len(complex_output))
  power = np.abs(complex_output)

  plt.plot(t, power)
  plt.show()

def main():
  init_params()
  process_arguments()
  complex_output = get_complex_output()
  # plt.plot(complex_output[300000:400000])
  # plt.show()
  process_output(complex_output)

if __name__ == '__main__':
  main()
