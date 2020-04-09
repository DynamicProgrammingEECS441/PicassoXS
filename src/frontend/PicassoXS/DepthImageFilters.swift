// THREE different filter
import UIKit

enum MaskParams {
  static let slope: CGFloat = 4.0
  static let width: CGFloat = 0.1
}

class DepthImageFilters {
  
  var context: CIContext
  
  init(context: CIContext) {
    self.context = context
  }
  
  init() {
    context = CIContext()
  }
  
  func createMask(for depthImage: CIImage, withFocus focus: CGFloat, andScale scale: CGFloat) -> CIImage {
    let s1 = MaskParams.slope
    let s2 = -MaskParams.slope
    let filterWidth =  2 / MaskParams.slope + MaskParams.width
    let b1 = -s1 * (focus - filterWidth / 2)
    let b2 = -s2 * (focus + filterWidth / 2)
    

    let mask0 = depthImage.applyingFilter("CIColorMatrix", parameters: [
    "inputRVector": CIVector(x: s1, y: 0, z: 0, w: 0),
    "inputGVector": CIVector(x: 0, y: s1, z: 0, w: 0),
    "inputBVector": CIVector(x: 0, y: 0, z: s1, w: 0),
    "inputBiasVector": CIVector(x: b1, y: b1, z: b1, w: 0)])
    .applyingFilter("CIColorClamp")

    let mask1 = depthImage
    .applyingFilter("CIColorMatrix", parameters: [
    "inputRVector": CIVector(x: s2, y: 0, z: 0, w: 0),
    "inputGVector": CIVector(x: 0, y: s2, z: 0, w: 0),
    "inputBVector": CIVector(x: 0, y: 0, z: s2, w: 0),
    "inputBiasVector": CIVector(x: b2, y: b2, z: b2, w: 0)])
    .applyingFilter("CIColorClamp")

    //combine two mask
    //using the CIDarkenBlendMode filter, which chooses the lower of the two values of the input masks.
    //Then  scale the mask to match the image size.
    let combinedMask = mask0.applyingFilter("CIDarkenBlendMode", parameters: ["inputBackgroundImage" : mask1])

    let mask = combinedMask.applyingFilter("CIBicubicScaleTransform", parameters: ["inputScale": scale])
    
    return mask
  }

  /*
    1. used the CIBlendWithMask filter and passed in the mask you created in the previous section. The filter essentially sets the alpha value of a pixel to the corresponding mask pixel value. So when the mask pixel value is 1.0, the image pixel is completely opaque and when the mask pixel value is 0.0, the image pixel is completely transparent. Since the UIView behind the UIImageView has a black color, black is what you see coming from behind the image.
    2.  create a CGImage using the CIContext.
    3. You then create a UIImage and return it.
  */
  func spotlightHighlight(image: CIImage, mask: CIImage, orientation: UIImageOrientation = .up) -> UIImage? {

    // 1
    let output = image.applyingFilter("CIBlendWithMask", parameters: ["inputMaskImage": mask])

    // 2
    guard let cgImage = context.createCGImage(output, from: output.extent) else {
      return nil
    }

    // 3
    return UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
  }

  // set the background image to be a greyscale version of the original image
  // This filter will show full color at the focal point based on the slider position and fade to grey from there.
  func colorHighlight(image: CIImage, mask: CIImage, orientation: UIImageOrientation = .up) -> UIImage? {

    let greyscale = image.applyingFilter("CIPhotoEffectMono")
    let output = image.applyingFilter("CIBlendWithMask", parameters: ["inputBackgroundImage" : greyscale,
                                                                      "inputMaskImage": mask])

    guard let cgImage = context.createCGImage(output, from: output.extent) else {
      return nil
    }

    return UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
  }


  func blur(image: CIImage, mask: CIImage, orientation: UIImageOrientation = .up) -> UIImage? {

  // 1
    let invertedMask = mask.applyingFilter("CIColorInvert")

  // 2
    let output = image.applyingFilter("CIMaskedVariableBlur", parameters: ["inputMask" : invertedMask,
                                                                         "inputRadius": 15.0])

  // 3
    guard let cgImage = context.createCGImage(output, from: output.extent) else {
      return nil
    }

  // 4
    return UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
  }


}


