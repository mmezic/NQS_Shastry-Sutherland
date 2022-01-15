9618663 test (with verbose)

9620396 - 8 ... korekce pro RBMSymm (je to ve slořce order_parameters) 
9622494 - 8 ... HARDER korekce pro RBMSymm (je to ve slořce order_parameters) 


projetí s ne-symm RBM
9618677 - 4
9618678 - 8 (wrong lattice)
9618679 - 8 (wrong lattice)

-- lol, bez symetrií to seběhlo strašně rychle
párty hárder:

9620392 - 8 (nyni uz spravna mrizka)
9620393f - 16
9620394f - 20
9620395f - 36
9624437 - 16
9624439 - 20
9624442 - 36
9638418 - 8 - zpřesňování mřížky výpočtem mezihodnot

Grid Searche:
9641942 - 8 - jen krátká zkouška, jeslti to nespadne
9641944 - 8
9641945 - 16
9641946 - 20

# GCNN
9882258 - 4  ... wrong MSR
9882259 - 8  ... wrong MSR
9882260 - 16 ... wrong MSR
9882262 - 20 ... wrong MSR
9882516 - 4
9882517 - 8

9882519 - 16
9882520 - 20 <-- just 2 layers; pouze 2 z (8,8,4), tedy asi (8,8)
9894504 - 8 second part of phase space
9885099 - 16 second part of phase space, because the first run probably does not have enough time
9894532 - 16 one missing point (J1=.5)
9895166 - 16 ... 2500 iterations instead of 1000 iterations which did not converge in the specific region
9885101 - 20 second part of phase space <-- just 2 layers of (8,8,4)

9894535 - 20 ... correctly 3 layerss (8,8,4)
9900156 - 16 ... 2500 iterations for points which did not converge with 1000 iterations (0,.5,.6,.666) <--- FAILED
9924351 - 16 ... 2500 iterations for points which did not converge with 1000 iterations (0,.5,.6,.666) <--- 2ND ATTEMPT => failed

TODO:

10031487 - 20 ... GCNN 3 layers (8,8,4) with MORE iterations (6000 instead of 2000) but less iters (500 instead of 1000)
10031496 - 16 ... RBM higher eta
10031499 - 16 ... RBM-1 lower alpha
10031501 - 16 ... RBM-2 TOTAL_SZ = 0
10031504 - 16 ... RBM-3 way lower eta
10031505 - 16 ... RBM-4 way higher eta

scp GCNN8_1.1_* mezic@tarkil8:~/diplomka/netket_scripts/convergence_plots