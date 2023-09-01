#### RNN VAE latent space
Reference: Gómez-Bombarelli, Rafael, et al. "Automatic chemical design using a data-driven continuous representation of molecules." ACS central science 4.2 (2018): 268-276.
 
 | PCA | latent space QED    | latent space SA
:-----:|:-------------------------:|:-------------------------:
|with classifier| ![](figures/QED_with_classifier.gif)  | ![](figures/SA.gif) |





 | PCA | latent space QED    | latent space SA
 :-----:|:-------------------------:|:-------------------------:
 | no classifier |  ![](figures/qed_no_classifier.png)  |  ![](figures/sa_no_classifier.png) |
 | with classifier |  ![](figures/qed_classifier.png)  |  ![](figures/sa_classifier.png) |




 | t-SNE | latent space QED, with classifier    | latent space SA, with classifier  
 :-----:|:-------------------------:|:-------------------------:
 | no classifier |  ![](figures/qed_no_classifier_tsne.png)  |  ![](figures/sa_no_classifier_tsne.png) |
 | with classifier |  ![](figures/qed_tsne.png)  |  ![](figures/sa_tsne.png) |


| reconstruction | no classifier    | with classifier  
 :-----:|:-------------------------:|:-------------------------:
| examples | ![](figures/recon_no_classifier.png) | ![](figures/recon.png) |
| sample quality | valid 91.967%, unique 91.967% | valid 92.500%, unique 92.500% | 
