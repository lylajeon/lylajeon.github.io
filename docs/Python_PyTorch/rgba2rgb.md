---
title: RGBA to RGB trouble shooting
layout: default
parent: Python, PyTorch
nav_order: 3
---
date: 2023-08-15



# RGBA to RGB Trouble shooting

## Pillow library is not reliable.

When changing tensor to Pillow's image, Pillow changes the range 0~1 to 0~255 automatically. 

It does not apply the real tensor value.

To apply the real tensor value, we have to use OpenCV.



## RGBA to RGB image 

- OpenCV, Pillow library both change RGBA image to RGB by simply getting rid of alpha channel. not applying alpha channel to R,G,B channels. I think this is a bug in OpenCV, Pillow library. 

- To correctly apply alpha channel to RGB channels, follow the details.

    - Normalise the RGBA values so that they're all between 0 and 1 - just divide each value by 255 to do this. We'll call the result `Source`.

    - Normalise also the matte colour (black, white whatever). We'll call the result `BGColor` **Note** - if the background colour is also transparent, then you'll have to recurse the process for that first (again, choosing a matte) to get the source RGB for this operation. (a black background - BGColor to 0,0,0. a white background - BGColor to 255,255,255.)

    - Now, the conversion is defined as (in complete psuedo code here!):

      ```
      Source => Target = (BGColor + Source) =
      Target.R = ((1 - Source.A) * BGColor.R) + (Source.A * Source.R)
      Target.G = ((1 - Source.A) * BGColor.G) + (Source.A * Source.G)
      Target.B = ((1 - Source.A) * BGColor.B) + (Source.A * Source.B)
      ```

    To get the final 0-255 values for `Target` you simply multiply all the normalised values back up by 255, making sure you cap at 255 if any of the combined values exceed 1.0 (this is over-exposure and there are more complex algorithms dealing with this that involve whole-image processing etc.).



- Actual Python code for changing RGBA tensor to RGB tensor

```python
# rgba_tensor : [batch_size, channel, height, width]
rgba_tensor = torch.mul(255,torch.randn(16, 4, 64, 256)) 
rgba_tensor = torch.clamp(rgba_tensor, 0, 255)

BG_color = (255,255,255) # white background
R_bg, G_bg, B_bg = BG_color

bs, c, h, w = rgba_tensor.size()
# rgb_tensor : rgba_tensor to RGB tensor 
rgb_tensor = torch.zeros(bs, 3, h, w)

r_ch, g_ch, b_ch, a_ch = rgba_tensor[:,0,:,:], rgba_tensor[:,1,:,:], rgba_tensor[:,2,:,:], rgba_tensor[:,3,:,:]
rem_a_ch = torch.sub(1., a_ch)

rgb_tensor[:,0,:,:] = torch.add(torch.mul(r_ch, a_ch), torch.mul(R_bg, rem_a_ch))
rgb_tensor[:,1,:,:] = torch.add(torch.mul(g_ch, a_ch), torch.mul(G_bg, rem_a_ch))
rgb_tensor[:,2,:,:] = torch.add(torch.mul(b_ch, a_ch), torch.mul(B_bg, rem_a_ch))
```



## tensor to numpy array

- opencv imwrite 하려면 BGR 순서로 바꿔줘야.
- image 로 표현하려면 범위체크 반드시! 0~1로 되어있다면 `*255` 필요

`np_arr = tensor.detach().to('cpu').numpy().transpose(1, 2, 0)`