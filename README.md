# Awesome-Training-Free-WAN2.1-Editing🍀  
**Star🌟 is a great help in open source development!**
  
*Awesome Training-Free methods meet WAN2.1-T2V.*  
It performs much better than [FlowDirector🦚](https://github.com/Westlake-AGI-Lab/FlowDirector)!!  

# Results🐦‍🔥
- Color/Background Editing
  
<table border="0" width="100%">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>WANAlign2.1</b></td>
  <td style="text-align:center;"><b>FlowDirector</b></td>
  <td style="text-align:center;"><b>WANEdit2.1</b></td>
</tr>
<tr>
  <td><video src="./videos/bear_832.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/background_color_editing/wanalign_brown_2_yellow.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/background_color_editing/flowdirector_brown_2_yellow.mp4" width="100%" controls autoplay loop></td>              
  <td><video src="./results/background_color_editing/wanedit_brown_2_yellow.mp4" width="100%" controls autoplay loop></td>
</tr>
<tr>
  <td style="text-align:center;" colspan="4">A large brown bear ... ➡️ A large yellow bear ...</td>
</tr>
<tr>
  <td><video src="./videos/snowboard_832.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/background_color_editing/wanalign_snowy_2_ocean.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/background_color_editing/flowdirector_snowy_2_ocean.mp4" width="100%" controls autoplay loop></td>              
  <td><video src="./results/background_color_editing/wanedit_snowy_2_ocean.mp4" width="100%" controls autoplay loop></td>
</tr>
<tr>
  <td style="text-align:center;" colspan="4">... in a snowy field. ➡️ ... in the ocean.</td>
</tr>
</table>

- Object Editing

<table border="0" width="100%">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>WANAlign2.1</b></td>
  <td style="text-align:center;"><b>FlowDirector</b></td>
  <td style="text-align:center;"><b>WANEdit2.1</b></td>
</tr>
<tr>
  <td><video src="./videos/bear_832.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/object_editing/wanalign_bear_2_tiger.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/object_editing/flowdirector_bear_2_tiger.mp4" width="100%" controls autoplay loop></td>              
  <td><video src="./results/object_editing/wanalign_bear_2_tiger.mp4" width="100%" controls autoplay loop></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A large brown bear ... ➡️ A large tiger ...</td>
</tr>
    
<tr>
  <td><video src="./videos/sea_turtle_832.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/object_editing/wanalign_turtle_2_seal.mp4" width="100%" controls autoplay loop></td>
  <td><video src="./results/object_editing/flowdirector_turtle_2_seal.mp4" width="100%" controls autoplay loop></td>              
  <td><video src="./results/object_editing/wanedit_turtle_2_seal.mp4" width="100%" controls autoplay loop></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A graceful sea turtle ... ➡️ A graceful seal ...</td>
</tr>
</table>

- Texture Editing

- Add Effect/Object


# Introduce WANAlign2.1🦖
```
(TODO)
```

# Quick Start (Code)🥏
## Environment
```
(TODO)
```

## Run code🏂
```
(TODO)
```

## Detail Code lines🏫
- FlowEdit Code: [WanPipelin.flowedit]()
- FlowAlign Code: [WanPipelin.flowalign]()
- Attention Extract Code: [wan_attention]()
- WANAlign Code: [WanPipelin.flowalign]()

# Paper Review✨
- [FlowEdit Review; korean]()
- [FlowDirector Review; korean]()
> Reviewed by kyujinpy🤗.

# TODO-list
- [ ] Paper Review 
- [x] Integrating Diffusers🤗
- [ ] Code Release
- [x] Support WAN2.1-T2V-1.3B
- [ ] Support WAN2.1-T2V-14B

# References
- [WAN2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- [MasaCtrl](https://github.com/TencentARC/MasaCtrl)
- [FlowEdit](https://matankleiner.github.io/flowedit/)
- [FlowAlign](https://arxiv.org/abs/2505.23145)
- [FlowDirector](https://arxiv.org/abs/2506.05046)
