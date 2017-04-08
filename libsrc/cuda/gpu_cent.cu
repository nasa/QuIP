/* special helper function for computing centroids */

// What is the strategy here?
// 
// We have two output arrays (for the weighted coordinates) and one input array
// with the values.  For each pixel, we compute the x and y coordinates from the
// thread index, and store the product of these with the pixel value.
// Later we will compute the sum...
//
// This could probably be composed as two vmul's followed by a vsum?


CK( type_code )

