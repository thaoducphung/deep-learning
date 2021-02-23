from torch import nn,optim 
import torchvision

model = torchvision.models.resnet18(pretrained=True)

# In finetuning, we freeze most of the model and 
# typically only modify the classifier layers to make predictions on new labels.
#  Let’s walk through a small example to demonstrate this. As before, we load 
#  a pretrained resnet18 model, and freeze all the parameters.

# Freeze all the parameters in the network
for param in model.parameters():
	param.requires_grad = False

model.fc = nn.Linear(512,10)

# Opitimize only the classifer
optimizer = optim.SGD(model.fc.parameters(),lr=1e-2,momentum=0.9)

