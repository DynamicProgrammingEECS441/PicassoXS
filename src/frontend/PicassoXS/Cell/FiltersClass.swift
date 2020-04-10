//
//  FiltersClass.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/22/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit


public class Filter{
    var FilterImage: UIImage
    //Tag to be implemented
    var name: String
    var intro: String
    
    init(_ Image: UIImage, _ Name: String, _ introduction: String){
        self.FilterImage = Image
        self.name = Name
        self.intro = introduction
    }
}
