from fsitk import fsitk



def test_biascorrection():
    ff = "/home/fraser/temp/test_bias_correction.vti"
    fsitk.biasCorrectionOfImageData_Generalised(ff, arrayName="PixelData", vtiFileOUT="/home/fraser/temp/temp.vti")




def main():
    test_biascorrection()


# ==================================================================
# S T A R T
#
if __name__ == '__main__':
    main()