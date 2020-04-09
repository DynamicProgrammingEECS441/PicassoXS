/*
1.First,  get a URL for an image file and safely type cast it to a CFURL.
2.then create a CGImageSource from this file.
3. From the image source at index 0, copy the disparity data (more on what that means later, but you can think of it as depth data for now) from its auxiliary data. The index is 0 because there is only one image in the image source. iOS knows how to extract the data from JPGs and HEIC files alike, but unfortunately this doesn’t work in the simulator.
4. prepare a property for the depth data. As previously mentioned, use AVDepthData to extract the auxiliary data from an image.
5. create an AVDepthData entity from the auxiliary data you read in.
6. ensure the depth data is the the format you need: 32-bit floating point disparity information.
7. Finally, return this depth data map.*/


import AVFoundation

struct DepthReader {
  
  var name: String
  var ext: String

  func depthDataMap() -> CVPixelBuffer? {

    // 1
    guard let fileURL = Bundle.main.url(forResource: name, withExtension: ext) as CFURL? else {
      return nil
    }

    // 2
    guard let source = CGImageSourceCreateWithURL(fileURL, nil) else {
      return nil
    }

    // 3
    guard let auxDataInfo = CGImageSourceCopyAuxiliaryDataInfoAtIndex(source, 0, 
        kCGImageAuxiliaryDataTypeDisparity) as? [AnyHashable : Any] else {
      return nil
    }

    // 4
    var depthData: AVDepthData

    do {
      // 5
      depthData = try AVDepthData(fromDictionaryRepresentation: auxDataInfo)

    } catch {
      return nil
    }

    // 6
    if depthData.depthDataType != kCVPixelFormatType_DisparityFloat32 {
      depthData = depthData.converting(toDepthDataType: kCVPixelFormatType_DisparityFloat32)
    }

    // 7
    return depthData.depthDataMap
  }

}


