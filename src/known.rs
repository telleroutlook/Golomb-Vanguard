/// Known optimal Golomb ruler lengths (OEIS A003022 / Shearer table).
/// Index = number of marks n, value = optimal ruler length L(n).
/// n >= 29 remains open problems.
pub const OGR_OPTIMAL: &[u32] = &[
    0,   // n=0 placeholder
    0,   // n=1
    1,   // n=2
    3,   // n=3
    6,   // n=4
    11,  // n=5
    17,  // n=6
    25,  // n=7
    34,  // n=8
    44,  // n=9
    55,  // n=10
    72,  // n=11
    85,  // n=12
    106, // n=13
    127, // n=14
    151, // n=15
    177, // n=16
    199, // n=17
    216, // n=18
    246, // n=19
    283, // n=20
    333, // n=21
    356, // n=22
    372, // n=23
    425, // n=24
    480, // n=25
    492, // n=26
    553, // n=27
    585, // n=28
];

pub fn optimal_length(n: usize) -> Option<u32> {
    OGR_OPTIMAL.get(n).copied()
}

pub fn required_words(max_bit: u32) -> usize {
    ((max_bit + 64) / 64) as usize
}
