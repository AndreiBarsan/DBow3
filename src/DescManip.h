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


/**
 * @brief Returns a string representation of an OpenCV Mat's type, like CV_8U.
 * @param type The type of a Mat, e.g., Mat.type().
 * @return A string representation of its type.
 */
std::string type2str(int type);


uint32_t DescManip::distance_8uc1(const cv::Mat &a, const cv::Mat &b) {
  //binary descriptor

  // Bit count function got from:
  // http://graphics.stanford.edu/~seander/bithacks.html#countbitssetkernighan

//  if (a.type() != CV_8UC1) {
//    throw std::runtime_error("The first descriptor's type must be CV_8UC1.");
//  }
//  if (b.type() != CV_8UC1) {
//    throw std::runtime_error("The first descriptor's type must be CV_8UC1.");
//  }
//  assert(a.cols == b.cols);
//  assert(a.rows == b.rows && a.rows == 1);
  // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0
//  assert(a.cols % sizeof(uint64_t) == 0);

  const uint64_t *pa, *pb;
  pa = a.ptr<uint64_t>(); // a & b are actually CV_8U
  pb = b.ptr<uint64_t>();
//  const uint8_t *pa, *pb;
//  pa = a.ptr<uint8_t>(); // a & b are actually CV_8U
//  pb = b.ptr<uint8_t>();

//  std::cout << "(" << a.rows << ", " << a.cols << "), (" << b.rows << ", " << b.cols << ")" << std::endl;

  uint64_t v, ret = 0;
  int n = a.cols / sizeof(uint64_t);
  // The vanilla version which doesn't rely on weird bit manipulations.
//  for (int i = 0; i < n; ++i, ++pa, ++pb) {
//    ret += __builtin_popcountl(pa[i] ^ pb[i]);
//  }
//  return ret;

  // This loop basically counts the number of bits that the two binary descriptors have in common.
  // We do it un chunks of 64 bits instead of 8 bits for performance reasons.
  for (size_t i = 0; i < n; ++i, ++pa, ++pb) {
    v = (*pa) ^ (*pb);
    v = v - ((v >> 1) & (uint64_t) ~(uint64_t) 0 / 3);
    v = (v & (uint64_t) ~(uint64_t) 0 / 15 * 3) + ((v >> 2) &
        (uint64_t) ~(uint64_t) 0 / 15 * 3);
    v = (v + (v >> 4)) & (uint64_t) ~(uint64_t) 0 / 255 * 15;
//    ret += v;   // meaningless; just for test
    ret += (uint64_t) (v * ((uint64_t) ~(uint64_t) 0 / 255)) >>
                                                             (sizeof(uint64_t) - 1) * CHAR_BIT;
  }
//  return ret;
  return static_cast<uint32_t>(ret);    // ret is a count, so definitely fits in 32 bits
}

} // namespace DBoW3

#endif
