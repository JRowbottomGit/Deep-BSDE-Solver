import numpy as np
# path = "./output_colab_ODE_ITO_decstep/"
path = ""

Trace = np.load(path + 'Trace.npy')
SHS = np.load(path + 'SHS.npy')


import pandas as pd
Trace_df = pd.DataFrame(Trace[:,:,0])
SHS_df = pd.DataFrame(SHS[:,:,0])
#     import pandas as pd
#     t_test_df = pd.DataFrame(t_test[:,:,0])
#     W_test_df = pd.DataFrame(W_test[:,:,0])

path = "SHS_output.xlsx"
with pd.ExcelWriter(path) as writer:
    Trace_df.to_excel(writer, sheet_name='Trace', index=False)
    SHS_df.to_excel(writer, sheet_name='SHS', index=False)
