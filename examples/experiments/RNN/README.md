#### RNN VAE latent space
Reference: Gómez-Bombarelli, Rafael, et al. "Automatic chemical design using a data-driven continuous representation of molecules." ACS central science 4.2 (2018): 268-276.
 
 | PCA | latent space QED    | latent space SA
:-----:|:-------------------------:|:-------------------------:
|with classifier| ![](figures/QED_recon.gif)  | ![](figures/SA_recon.gif) |
| prediction | ![](figures/qed_pred.png) | ![](figures/sa_pred.png) |




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
| examples | ![](figures/recon_no_classifier.png) | ![](figures/recon_classifier.png) |
| sample quality | valid 91.967%, unique 91.967% | valid 92.500%, unique 92.500% | 


#### deficit: using char to tokenize SMILES. Hard to recon exactly and smiles may not be valid
| category | SMILES
:-------------------------|:-------------------------
|Fedratinib | Cc1cnc(nc1Nc1cccc(c1)S(=O)(=O)NC(C)(C)C)Nc1ccc(cc1)OCCN1CCCC1
pred_z  smi | Cc1ccc(Nc2NC2ccc((C2)[(=O)(=O)C((=)CC)C
pred_mu smi | Cc1ccc(Nc2NC2ccc((C2)[(=O)(=O)C((=)CC)C
