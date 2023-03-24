## **Emotion Detection: Whether emotion is sincere or not.**

*Steps*: 
* Open the .py file, either you can detect emotions via camera (cam()) or in an image(image()).
* Press 'q' to stop the run if you are reunning the cam() function.
* It works on the NEW model, if some landmarks are not detected it revert backs to the OLD model.

# **Validation accuracy achieved is in the range(76-81) **

**Model Files Link(.h5)** : https://drive.google.com/file/d/1wzZj0v1aT0c88d3Pml-DKX-3k7ASRRVk/view?usp=sharing(NEW)
https://drive.google.com/file/d/1LH52IiHv4qpYdtIn5zuDpF31GrwHCjeK/view?usp=share_link(OLD)
{both are used in the .py file}

# *Note:*Also for the frontal face detection I have used use haarcascade, which I have uploaded.

Determining the sincerity of an emotion is a complex and nuanced process, as emotions can be expressed in different ways depending on the individual and the situation. The following can help us in determining the sincerity of an emotion:

# *1) Looking for consistency*: If someone's emotional expression is consistent with their words and actions, it's more likely to be sincere. For example, if someone expresses happiness and their body language and behavior reflect that emotion, it's more likely to be genuine.

# *2) Considering context*: Emotions can be influenced by a range of factors, such as culture, personality, and past experiences. Consider the context in which the emotion is expressed, as well as the person's background and history, to better understand the sincerity of their emotions.

# *3) Looking for microexpressions*: Microexpressions are brief, involuntary facial expressions that reveal an individual's true emotions. If someone is trying to hide their emotions, they may still display microexpressions that reveal their true feelings.

# *4) Listening to the tone of voice*: The tone of voice can provide clues to the sincerity of an emotion. If someone's voice sounds flat or robotic, it may suggest that they are not being genuine.

Below we have worked upon the third point i.e. **looked for microexpressions** in an image. We started by extracting micro features from the images, as such features can somewhat give us a cue as to the state of the person emotions, such as:

* *Eyebrow position*: When a person is surprised or angry, their eyebrows are typically raised. When they are sad or disappointed, their eyebrows may be lowered.

* *Mouth shape*: The shape of a person's mouth can indicate a range of emotions. For example, a smile may indicate happiness, while a frown may indicate sadness or anger.

* *Wrinkles around the eyes*: The presence of wrinkles around the eyes, also known as crow's feet, can indicate that a person is smiling or happy.

* *Eye openness*: The size of a person's eyes can indicate a range of emotions. For example, when a person is surprised or scared, their eyes may widen. Conversely, when a person is angry, their eyes may narrow.

* *Lip curvature*: The curvature of a person's lips can indicate whether they are smiling or frowning.

* *Nose wrinkles*: The presence of wrinkles on a person's nose can indicate a range of emotions, such as disgust or anger.

* *Jaw tension*: The tension in a person's jaw muscles can provide an indication of their emotional state. For example, a person who is angry or stressed may clench their jaw.

* *Brow furrows*: The degree to which a person's eyebrows are furrowed can indicate their level of stress or anger.

* *Chin thrust*: The position of a person's chin can indicate their level of confidence or assertiveness.

# **PS** : In total we have extracted 18 features and used them in addition to the features extracted from our images.

# * Furthermore, 

#  1) We can move towards **body postures** and the information it conveys and, 

#  2) Parsing through a person's speech via **speech analysis** algorithms that can detect changes in pitch, tone, and other characteristics that may indicate sincerity or insincerity.

Combining all the above will surely give us a model which detects emotions, further classifying whether it's sincere or not.