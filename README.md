# Audio Style Transfer

<p align="center"> <i>Audio Style Transfer</i> aims to create unique audio samples by combining a base audio file and style audio file. We do this by applying a computer vision techique known as "style transfer". We also propose two preprocessing and an alternate dimensional representation of sound to acheive preferred results.</p> Paper is available at: https://drive.google.com/file/d/1wXY5o-cbB8-CToxqB5jCDXJz-AaAiptw/view?usp=sharing


<p align="center"> <b>Eric Chang, Mark Endo, John Guibas</b> </p>


# Style Tranfer

*Style tranfer* requires two different files: a content and a style. The style is extracted from the file and applied to the content. In this case, the content is the notes of the content file, while the style is how the style file is played, such as rhythm or instruments. The objective of the style transfer is to get the program to play the content audio file in the style extracted from the style audio file.

# Pitch Shift
This method shifts each pitch of the style song to match that of the content song. The pitches of the style song are changed because pitches are an aspect of content, not style. In order to implement this method, each audio clip is split into small fragments. We calculate the pitch for each fragment using parabolic interpolation to find the peak in a spectrogram graph, which in this case represents the greatest magnitude pitch in the audio fragment. Once all pitches are detected, the pitch of each style fragment is changed to match the pitch of the corresponding content fragment.The final step is combining each of the fragments together. However, playing the new style audio without any further manipulation results in a fast clipping sound, somewhat like the firing of a gun. This unwanted effect is caused by the audio fragments being out of phase due to overlapping audio fragments. To counteract this phase problem, we recombined the overlapping audio fragments with a triangle window. By manipulating the frame size and hop size, we were able to achieve a smooth transition between each audio fragment.


# Guided Music Synthesis via Markov Random Oracle

This method resolves the time disparities between the segments of the content and style audio. For example, an unedited segment of the style audio may have rests while the content audio is playing a note. When run through the neural network, the style for that frame will not have a reading, and the output will be a distorted sounding version of the content note. Therefore, the style audio needs to be edited in a way that creates a corresponding sound for each content sound. When using the VMO method, a target file is shuffled around to better match a query audio file. Originally, this was used in jazz machine improvisation with a target solo audio file and a query accompaniment audio file. However, we realized that we could use this to fit the timing of a style audio file to a content audio file. In this case, the style audio is the target and the content audio is the query. Although the style timing is being altered, this has no effect on the style quality because we do not consider timing as an aspect of style. Now when run through the neural network, each content sound will have a corresponding style sound. 

