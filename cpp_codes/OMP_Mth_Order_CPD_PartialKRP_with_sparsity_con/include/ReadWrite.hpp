#ifndef READ_WRITE_HPP
#define READ_WRITE_HPP

#include "master_lib.hpp"
#include <string>

#include <iostream>
#include <fstream>
// Open Function Sys Call
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
// Read-Write Functions Sys Call
#include <unistd.h>

/*------------------------------------------------------------------------------------------------*/
/*--+ Read functions +--*/
/*------------------------------------------------------------------------------------------------*/

/* 
 * void readEigenMatrix(<OUT>  Ref<MatrixXd> EigenData,
 *                      <IN>   std::ifstream &file)
 * 
 * Description: Read DATA of type <Eigen::MatrixXd> from an input file.
 */
void readEigenMatrix( Ref<MatrixXd> EigenData, 
                      std::ifstream &file)
{
    MatrixXd DataFromFile(EigenData.cols(), EigenData.rows());
    file.read((char *)DataFromFile.data(), DataFromFile.rows() * DataFromFile.cols() * sizeof(double));
    // Data in file are stored in ROW-MAJOR order. Eigen uses COL-MAJOR order.
    // Convert ROW-MAJOR to COL-MAJOR.
    // #ifdef FACTORS_ARE_TRANSPOSED
    // EigenData = DataFromFile;
    // #else
    EigenData = DataFromFile.transpose();
    // #endif
}

/* 
 * void readEigenDataFromFile( <IN>   const std::string fileName
 *                             <OUT>  Ref<MatrixXd> EigenData)
 * 
 * Description: Check fileName and read DATA using function readEigenMatrix(...) .
 */
void readEigenDataFromFile( const std::string fileName,
                            Ref<MatrixXd>     EigenData )
{
    std::ifstream is;
    try
    {
        // Open FILE.
        is.open(fileName.c_str(), std::ios::in | std::ios::binary);

        if (is.fail())
        {
            std::cerr << "Input file: " << fileName << " not found!" << std::endl;
            std::cerr << "Terminating..." << std::endl;
            exit(1);
            // throw -1;
        }

        readEigenMatrix(EigenData, is);

        is.close();
    }
    catch (std::ofstream::failure const &ex)
    {
        std::cerr << "Exception opening/reading/closing file: " << fileName << std::endl;
    }
}

/*------------------------------------------------------------------------------------------------*/
/*--+ Write functions +--*/
/*------------------------------------------------------------------------------------------------*/

/* 
 * void writeEigenMatrix(<IN>   Ref<MatrixXd> EigenData,
 *                       <OUT>  std::ifstream &file)
 * 
 * Description: Write DATA of type <Eigen::MatrixXd> to an output file.
 */
void writeEigenMatrix(const Ref<const MatrixXd> EigenData, std::ofstream &file)
{
    // #ifdef FACTORS_ARE_TRANSPOSED
    // file.write((char *)EigenData.data(), EigenData.rows() * EigenData.cols() * sizeof(double));
    // #else
    MatrixXd DataFromFile = EigenData.transpose();
    // Data in file are stored in ROW-MAJOR order. Eigen uses COL-MAJOR order.
    // Convert COL-MAJOR to ROW-MAJOR.
    file.write((char *)DataFromFile.data(), DataFromFile.rows() * DataFromFile.cols() * sizeof(double));
    // #endif
}

/* 
 * void writeEigenDataToFile( <IN>   const std::string fileName
 *                            <IN>  Ref<MatrixXd> EigenData)
 * 
 * Description: Check fileName and write DATA using function writeEigenMatrix(...) .
 */
void writeEigenDataToFile( const std::string fileName,
                           Ref<MatrixXd> EigenData )
{
    std::ofstream os;
    try
    {
        // Open FILE.
        os.open(fileName.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);

        writeEigenMatrix(EigenData, os);

        os.close();
    }
    catch (std::ofstream::failure const &ex)
    {
        std::cerr << "Exception opening/reading/closing file: " << fileName << std::endl;
    }
}

#endif