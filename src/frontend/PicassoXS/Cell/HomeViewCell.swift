//
//  HomeViewCell.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/23/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit


class HomeViewCell: UITableViewCell{
    
    @IBOutlet weak var ViewContainer: UIView!
    @IBOutlet weak var ResultIm: UIImageView!
    @IBOutlet weak var TimeStamp: UILabel!
    @IBOutlet weak var ClippingView: UIView!
    
    func setHomeView(Image: MadeImages){
        ResultIm.image = Image.res
        TimeStamp.text = Image.DateTime
        
        
        
        ClippingView.layer.cornerRadius = 20
        ClippingView.backgroundColor = GlobalVariables.White
        ClippingView.layer.borderColor = GlobalVariables.Grey1.cgColor
        ClippingView.layer.borderWidth = 0.5
        ClippingView.layer.masksToBounds = true
        
        ViewContainer.layer.cornerRadius = 20
        ViewContainer.layer.shadowOpacity = 0.05
        ViewContainer.layer.shadowRadius = 10
//        ViewContainer.layer.shadowColor = UIColor(named: "Orange")?.cgColor
        ViewContainer.layer.shadowOffset = CGSize(width: 3, height: 3)
        ViewContainer.backgroundColor = UIColor(named: "Red")
        
        
        
    }
    
    
}
