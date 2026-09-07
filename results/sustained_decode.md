# sustained decode: 5 back-to-back tg1024 runs, NPU OPPOLL, no cooldown; max thermal zone after each. 02:10
run 1  tg1024   30.05 ± 0.00 
  temp_max=95700
run 2  tg1024   29.93 ± 0.00 
  temp_max=94600
run 3  tg1024   29.85 ± 0.00 
  temp_max=95300
run 4  tg1024   29.83 ± 0.00 
  temp_max=98000
run 5  tg1024   29.82 ± 0.00 
  temp_max=97200
SUSTAINED_DONE

# control: same 5 runs WITHOUT OPPOLL (interrupt mode), started at temp_max=59600 02:15
run 1  tg1024   26.44 ± 0.00 
  temp_max=79600
run 2  tg1024   26.48 ± 0.00 
  temp_max=79600
run 3  tg1024   26.46 ± 0.00 
  temp_max=81100
run 4  tg1024   26.44 ± 0.00 
  temp_max=81900
run 5  tg1024   26.47 ± 0.00 
  temp_max=83000
