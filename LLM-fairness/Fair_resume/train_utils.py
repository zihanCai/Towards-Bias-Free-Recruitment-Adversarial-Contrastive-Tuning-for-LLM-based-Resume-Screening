
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pretrain_main_task(model, optimizer_main, train_loader, loss_criterion, epochs):

    pretrain_main_loss = 0
    steps = 0

    for epoch in range(epochs):

        # print("Epoch: ", epoch + 1)
        epoch_loss = 0
        epoch_batches = 0
        loop = tqdm((train_loader), total = len(train_loader),leave =False)

        for batch in loop: # starting from the 0th batch
            # get the inputs and labels
            resumes = batch['resumes']
            descriptions = batch['descriptions']
            bias_labels = batch["labels"].to(device) 

            optimizer_main.zero_grad()

            resume_features = model.tokenize(resumes)
            job_desc_features = model.tokenize(descriptions)

            resume_features = {key: val.to(device) for key, val in resume_features.items()}
            job_desc_features = {key: val.to(device) for key, val in job_desc_features.items()}

            sentence_features = [resume_features, job_desc_features]
            # resume_embeddings = model(resume_features)["sentence_embedding"]
            # job_desc_embeddings = model(job_desc_features)["sentence_embedding"]
        
            main_loss = loss_criterion(sentence_features,labels = None)

            main_loss.backward() # back prop
            optimizer_main.step()

            loop.set_description(f'Pretrain Main Task----Epoch [{epoch}/{epochs}]')

            pretrain_main_loss += main_loss.item()
            epoch_loss += main_loss.item()
            epoch_batches += 1
            steps += 1

            loop.set_postfix(loss = main_loss.item())


        print("Average Pretrain Classifier epoch loss: ", epoch_loss/epoch_batches)
    print("Average Pretrain Classifier batch loss: ", pretrain_main_loss/steps)

    return model


def pretrain_adversary(adv, model, optimizer_adv, train_loader, loss_criterion, epochs):
  
    pretrain_adversary_loss = 0
    steps = 0

    for epoch in range(epochs):
        #print("Epoch: ", epoch + 1)
        epoch_loss = 0
        epoch_batches = 0
        loop = tqdm((train_loader), total = len(train_loader),leave =False)

        for batch in loop: # starting from the 0th batch    
            # get the inputs and labels
            resumes = batch['resumes']
            descriptions = batch['descriptions']
            bias_labels = batch["labels"].to(device)  

            resume_features = model.tokenize(resumes)
            resume_features = {key: val.to(device) for key, val in resume_features.items()}
        
            optimizer_adv.zero_grad()
            resume_embeddings = model(resume_features)["sentence_embedding"]
            adversary_output = adv(resume_embeddings)
            adversary_loss = loss_criterion(adversary_output, bias_labels.long()) # compute loss
            adversary_loss.backward() # back prop
            optimizer_adv.step()

            loop.set_description(f'Pretrain Adversary----Epoch [{epoch}/{epochs}]')
            pretrain_adversary_loss += adversary_loss.item()
            epoch_loss += adversary_loss.item()
            epoch_batches += 1
            steps += 1
            loop.set_postfix(loss = adversary_loss.item())

        print("Average Pretrain Adversary epoch loss: ", epoch_loss/epoch_batches)
    print("Average Pretrain Adversary batch loss: ", pretrain_adversary_loss/steps)

    return adv


def train_adversary(adv, model, optimizer_adv, train_loader, loss_criterion, epochs=1):
  
    adv_loss = 0
    steps = 0

    for epoch in range(epochs):
        loop = tqdm((train_loader), total = len(train_loader),leave =False)
        for batch in loop: # starting from the 0th batch
            # get the inputs and labels
            resumes = batch['resumes']
            descriptions = batch['descriptions']
            bias_labels = batch["labels"].to(device)  

            resume_features = model.tokenize(resumes)
            resume_features = {key: val.to(device) for key, val in resume_features.items()}
        
            optimizer_adv.zero_grad()

            resume_embeddings = model(resume_features)["sentence_embedding"]

            adversary_output = adv(resume_embeddings)
            adversary_loss = loss_criterion(adversary_output, bias_labels.long()) # compute loss
 
            adversary_loss.backward() # back prop
            optimizer_adv.step()

            loop.set_description(f'Train Adversary----Epoch [{epoch}/{epochs}]')
            adv_loss += adversary_loss.item()
            steps += 1
            loop.set_postfix(loss = adversary_loss.item())
  
    print("Average Adversary batch loss: ", adv_loss/steps)

    return adv

def train_main_task(model, optimizer_main, adv, train_loader, adv_loss_criterion, main_loss_criterion,lbda):


    loop = tqdm((train_loader), total = len(train_loader),leave =False)
    for batch in loop: # starting from the 0th batch
        # get the inputs and labels
        resumes = batch['resumes']
        descriptions = batch['descriptions']
        bias_labels = batch["labels"].to(device) 


        # Toxic classifier part

        optimizer_main.zero_grad()

        resume_features = model.tokenize(resumes)
        job_desc_features = model.tokenize(descriptions)

        resume_features = {key: val.to(device) for key, val in resume_features.items()}
        job_desc_features = {key: val.to(device) for key, val in job_desc_features.items()}

        sentence_features = [resume_features, job_desc_features]
        resume_embeddings = model(resume_features)["sentence_embedding"]

        adversary_output = adv(resume_embeddings)
        adversary_loss = adv_loss_criterion(adversary_output, bias_labels.long())

        main_loss = main_loss_criterion(sentence_features, labels=None) # compute loss

        total_loss = main_loss - lbda * adversary_loss
        total_loss.backward() # back prop
      
        optimizer_main.step()

        loop.set_description(f'Train Main+Adversary')
        loop.set_postfix(main_loss = main_loss.item(),adv_loss = adversary_loss.item(),total_loss=total_loss.item())
        # print("Adversary Mini-Batch loss: ", adversary_loss.item())
        # print("Main task Mini-Batch loss: ", main_loss.item())
        # print("Total Mini-Batch loss: ", total_loss.item())
        break
    return model