```
task                               dim            seeds      SR    ref      d   Score    ref      d  per-seed SR ours | ref
---------------------------------------------------------------------------------------------------------------------------
arrange_largest_number             generalization -         -      -      -       -      -      -    - | 0/0/2
fold_clothes                       generalization -         -      -      -       -      -      -    - | 54/50/62
hang_mugs                          generalization -         -      -      -       -      -      -    - | 8/6/2
make_toast                         generalization -         -      -      -       -      -      -    - | 0/6/2
pack_objects_into_box              generalization -         -      -      -       -      -      -    - | 2/6/6
pour_liquid_into_cup               generalization -         -      -      -       -      -      -    - | 36/40/28
push_T                             generalization -         -      -      -       -      -      -    - | 0/0/0
sort_nesting_dolls_by_size         generalization -         -      -      -       -      -      -    - | 4/4/6
stack_blocks                       generalization -         -      -      -       -      -      -    - | 8/4/10
stack_bowls                        generalization -         -      -      -       -      -      -    - | 46/46/44
store_laptop_and_headphones        generalization -         -      -      -       -      -      -    - | 16/20/12
sweep_blocks                       generalization -         -      -      -       -      -      -    - | 0/4/0
classify_objects                   long-horizon   -         -      -      -       -      -      -    - | 0/0/4
fill_egg_holder                    long-horizon   -         -      -      -       -      -      -    - | 0/0/0
fill_pen_holder                    long-horizon   -         -      -      -       -      -      -    - | 2/10/6
make_kong                          long-horizon   -         -      -      -       -      -      -    - | 30/36/30
organize_table                     long-horizon   -         -      -      -       -      -      -    - | 16/8/16
play_stacking_toy                  long-horizon   -         -      -      -       -      -      -    - | 0/0/0
play_tic_tac_toe                   long-horizon   -         -      -      -       -      -      -    - | 54/52/70
put_bottles_into_dustbin           long-horizon   0       96.00  98.00  -2.00   96.80  98.00  -1.20  96 | 98/88/88
cover_blocks                       memory         -         -      -      -       -      -      -    - | 16/12/22
imitate_sorting_sequence           memory         -         -      -      -       -      -      -    - | 2/0/0
match_and_pick_from_conveyor       memory         -         -      -      -       -      -      -    - | 42/38/30
press_by_number                    memory         -         -      -      -       -      -      -    - | 0/0/0
swap_T                             memory         -         -      -      -       -      -      -    - | 0/0/0
swap_blocks                        memory         -         -      -      -       -      -      -    - | 0/2/0
align_blocks                       open           -         -      -      -       -      -      -    - | 0/0/0
classify_objects_by_language       open           -         -      -      -       -      -      -    - | 0/0/0
general_pickup                     open           -         -      -      -       -      -      -    - | 6/8/12
pick_from_conveyor_by_image        open           -         -      -      -       -      -      -    - | 0/0/0
pour_by_language                   open           -         -      -      -       -      -      -    - | 0/0/0
solve_equation                     open           -         -      -      -       -      -      -    - | 0/0/0
stack_blocks_by_language           open           -         -      -      -       -      -      -    - | 0/0/0
store_tools_in_toolbox             open           -         -      -      -       -      -      -    - | 0/0/0
build_tower                        precision      -         -      -      -       -      -      -    - | 38/42/36
deposit_coin                       precision      -         -      -      -       -      -      -    - | 2/2/6
fasten_screws                      precision      -         -      -      -       -      -      -    - | 6/4/2
insert_key                         precision      -         -      -      -       -      -      -    - | 0/0/0
insert_tubes                       precision      -         -      -      -       -      -      -    - | 8/4/8
play_Xylophone                     precision      -         -      -      -       -      -      -    - | 0/0/0
plug_in_charger                    precision      -         -      -      -       -      -      -    - | 0/4/8
pour_balls_into_vase               precision      -         -      -      -       -      -      -    - | 16/16/20

cells finished: 1; tasks with a finished seed: 1/42; tasks with all 3 seeds: 0/42; drivers: 580.159.04
long-horizon   tasks  1/ 8  SR  96.00 vs ref  98.00   Score  96.80 vs ref  98.00   published 25.33/34.93  (partial: reference over the same cells)
average        dims  1/5   SR  96.00 vs ref  98.00   Score  96.80 vs ref  98.00   published 11.92/17.18  (partial: reference over the same cells)
```
