# Transform Photos to Monet Paintings with CycleGANs

Artists like Claude Monet are recognized for the unique styles of their works, such as the unique colour scheme and brush strokes. These are hard to be imitated by normal people, and even for professional painters, it will not be easy to produce a painting whose style is Monet-esque. However, thanks to the invention of Generative Adversarial Networks (GANs) and their many variations, Data Scientists and Machine Learning Engineers can build and train deep learning models to bring an artist's peculiar style to your photos. 



<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0KSEEN/images/starry_night.png" width="60%">

## Table of Contents

<ol>
    <li><a href="https://#Objectives">Objectives</a></li>
    <li>
        <a href="https://#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
            <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
            <li><a href="#Defining Helper Functions">Defining Helper Functions</a></li>
        </ol>       
    </li>
    <li><a href="#What is Image Style Transfer in Deep Learning?">What is Image Style Transfer in Deep Learning?</a></li>
    <li><a href="#CycleGANs">CycleGANs</a>
        <ol>
            <li><a href="#A quick recap on vanilla GANs">A quick recap on vanilla GANs</a></li>
            <li><a href="#What's novel about CycleGANs?">What's novel about CycleGANs?</a></li>
        </ol>  
    </li>   
    <li><a href="#Data Loading">Data Loading</a></li>
    <li><a href="#Building the Generator">Building the Generator</a>
        <ol>
            <li><a href="#Defining the Downsampling Block">Defining the Downsampling Block</a></li>
            <li><a href="#Defining the Upsampling Block">Defining the Upsampling Block</a></li>
            <li><a href="#Assembling the Generator">Assembling the Generator</a></li>
        </ol>  
    </li>   
    <li><a href="#Building the Discriminator">Building the Discriminator</a></li>
    <li><a href="#Building the CycleGAN Model">Building the CycleGAN Model</a>
    <li><a href="#Defining Loss Functions">Defining Loss Functions</a> 
    <li><a href="#Model Training">Model Training</a>  
         <ol>
            <li><a href="#Training the CycleGAN">Training the CycleGAN</a></li>
        </ol>  
    </li>     
    <li><a href="#Visualize our Monet-esque photos">Visualize our Monet-esque photos</a>
        <ol>
            <li><a href="#Loading the Pre-trained Weights">Loading the Pre-trained Weights</a></li>
            <li><a href="#Visualizing Style Transfer Output">Visualizing Style Transfer Output</a></li>
        </ol>       
    </li>
</ol>
