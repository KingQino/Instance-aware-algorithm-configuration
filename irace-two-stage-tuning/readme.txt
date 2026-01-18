The directory indicates how we get the instance-specific labels using two-stage parameter tuning through IRACE. 

1. a coarse-grained tuning is first applied. 
2. the multimodal search space for each parameter is detected, unimodel otherwise.  
3. for each mode (multimodal per instance maybe), determine the tuning range of each parameter in the fine-grained stage. ("Refine-Range.ipynb")
4. a fine-grained stage is applied. Finally, collect median and representative statistics from the top-K fine-grained tuning results.

The obtained top-K parameter configurations in the fine-grained stage are what we need for the next step. 


In the fine-grained stage, we have to manually tune the "optimal" parameter configurations for the 4 hardest instances due to the maximum walltime limitation of our computing resourcess (Sulis HPC limits a maximum 48 hours runing). 
The 4 hardest instances are listed below:
- X-n819-k171.evrp
- X-n916-k207.evrp
- X-n830-k171-s11.evrp
- X-n920-k207-s4.evrp
