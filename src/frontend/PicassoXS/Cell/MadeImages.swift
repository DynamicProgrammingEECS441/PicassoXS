//
//  HomeImage.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/21/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit




//Create a instance of an image created
public class MadeImages{
    var DateTime: String
    var input: UIImage
    var filter: UIImage
    var res: UIImage
    
    init(_ inputIma: UIImage, _ filterIma: UIImage, _ result: UIImage){
        self.input = inputIma
        self.filter = filterIma
        self.res = result
        
        //Set Date
        let df = DateFormatter()
        df.dateFormat = "EEEE, MMM d, yyyy"
        DateTime = df.string(from: Date())
        
    }
}


extension Date {
    static func getFormattedDate(date: Date, format: String) -> String {
        let dateformat = DateFormatter()
        dateformat.dateFormat = format
        return dateformat.string(from: date)
    }
}


