<h1 align="center"> Defensive Adversarial Mix-Up (DAM) Neural Network Models </h1>
<p align="center">Augmenting for the best and most efficient image classification model.</p>
<br>

## P R O J E C T &nbsp;&nbsp;&nbsp; O V E R V I E W
  In this section, we discuss the architecture of the Defensive Adversarial Mix-Up (DAM) Neural Network and the individual components that make this technique novel. Image  filters,  image  Mix-Up,  and  Defensive  Adversarial Mix-up are the three main components of the DAMN strategy. The below figure depicts the DAM technique’s modular architecture. 

<p align="center"><img src="https://github.com/imohammad12/Mixing-Augmentation/blob/main/CS698_Architecture.png" width="500" height="700"></p> 

  Image  filters  is  the  first  component  of  the  model  and  is divided into two sub-modules: attention filtering and environ-mental  filtering.  The  aim  of  attention  filters  is  to  highlight the  important  sections  of  the  input  image,  such  as  the  targetobject, while fading out the surrounding regions. Environmentalfilters,  in  contrast  to  attention  filters,  aim  to  augment  several surrounding scenarios for the target object, and it is performed in  three  variants:  occlusion  scenarios,  road  conditions,  and weather  conditions.  The  goal  of  filtering  is  to  aid  the  neural network  model  in  determining  the  object’s  importance  in relation to its surroundings. In  the  image  Mix-up  portion  of  our  model  we  perform four  different  types  of  mix-ups  after  acquiring  the  filteredimages:  Linear  Mix-Up,  Vertical  Concatenations, and HorizontalConcatenations. The algorithms for performing the mix-ups are discussed in detail in the paper, and the strategies used for defensive adversarial training for each of the mix-ups are also detailed.
