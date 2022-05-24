# prevent two theards to go into critical section one after the other

condition = True

# one output, will indicate condition

threadA{


    if condition:
       x = 0
        condition = False
        csa


    else:
        condition = True





}


threadB{









    if condition:
        csb
        condition = False
    else:
        condition = True




}