![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)


# Project: Automatic Emergency Braking (AEB): Build ML models to decide braking 

### Goal 
- Automatically decide when to apply braking to avoid ormitigate collision using distance and realtive motion. 
#### Inputs:
- Ego speed ($v_e$)
- Lead speed ($v_i$)
- Distance gap ($d$)
- Relative speed: $v_r=v_r$ (if $v_r>0$)
#### Time to Collision (TTC): $TTC = \frac{d}{v_r}$ if ($v_r>0$)
#### Trigger Rule:
- if TTC < 2 seconds $\rightarrow$ Brake 
- OR if d < 10m $\rightarrow$ Brake