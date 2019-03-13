/**
 * File: FClass.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: generic FClass to instantiate templated classes
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_DESCMANIP__
#define __D_T_DESCMANIP__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include "exports.h"

namespace DBoW3 {

/// Class to manipulate descriptors (calculating means, differences and IO routines)
class DBOW_API DescManip
{
public:
  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
   static void meanValue(const std::vector<cv::Mat> &descriptors,
    cv::Mat &mean)  ;
  
  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
   static double distance(const cv::Mat &a, const cv::Mat &b);
   static  inline uint32_t distance_8uc1(const cv::Mat &a, const cv::Mat &b);

  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const cv::Mat &a);
  
  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(cv::Mat &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<cv::Mat> &descriptors,
    cv::Mat &mat);

  /**io routines*/
  static void toStream(const cv::Mat &m,std::ostream &str);
  static void fromStream(cv::Mat &m,std::istream &str);
public:
  /**Returns the number of bytes of the descriptor
   * used for binary descriptors only*/
  static size_t getDescSizeBytes(const cv::Mat & d){return d.cols* d.elemSize();}
};

uint32_t DescManip::distance_8uc1(const cv::Mat &a, const cv::Mat &b) {
  //binary descriptor

  if (a.type() != CV_8UC1) {
    throw std::runtime_error("The first descriptor's type must be CV_8UC1.");
  }
  if (b.type() != CV_8UC1) {
    throw std::runtime_error("The first descriptor's type must be CV_8UC1.");
  }
  assert(a.cols == b.cols);
  assert(a.rows == b.rows && a.rows == 1);
  // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0
  assert(a.cols % sizeof(uint64_t) == 0);

  const uint64_t *pa, *pb;
  pa = a.ptr<uint64_t>(); // a & b are actually CV_8U, but we process 4 x 8 bits at a time!
  pb = b.ptr<uint64_t>();
  int n = a.cols / sizeof(uint64_t);
  uint64_t ret = 0;
  uint64_t v = 0;

  // This loop basically counts the number of bits that the two binary descriptors have in common.
  // We do it un chunks of 64 bits instead of 8 bits for performance reasons.
  for (size_t i = 0; i < n; ++i, ++pa, ++pb) {
    v = (*pa) ^ (*pb);
    v = v - ((v >> 1) & (uint64_t) ~(uint64_t) 0 / 3);
    v = (v & (uint64_t) ~(uint64_t) 0 / 15 * 3) + ((v >> 2) &
        (uint64_t) ~(uint64_t) 0 / 15 * 3);
    v = (v + (v >> 4)) & (uint64_t) ~(uint64_t) 0 / 255 * 15;
    ret += (uint64_t) (v * ((uint64_t) ~(uint64_t) 0 / 255)) >>
                                                             (sizeof(uint64_t) - 1) * CHAR_BIT;
  }
  return static_cast<uint32_t>(ret);

//  uint32_t ret = 0;
//  int n = a.cols / sizeof(uint64_t);
//
//  // A clean, fast implementation I will NOT use for the deadline, just in case.
//  // Loop through every 64-bit chunk in the descriptors...
//  for (int i = 0; i < n; ++i) {
//    // ...and count the number of bits they have in common!
//    // This GCC intrinsic does it for us in one go!
//    ret += __builtin_popcountl(pa[i] ^ pb[i]);
//  }
//
//  // An old bug here would return the 64-bit value as 32-bit, causing a SEGFAULT.
//  return ret;

}

} // namespace DBoW3

#endif
