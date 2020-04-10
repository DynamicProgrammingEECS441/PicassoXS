//
//  SettingCell.swift
//  PicassoXFinal
//
//  Created by Senhao Wang on 4/9/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit


class BaseCell: UICollectionViewCell
{
    override init(frame: CGRect) {
        super.init(frame: frame)
        setUpView()
    }

    func setUpView()
    {

    }
    required init?(coder aDecoder: NSCoder) {
        fatalError("init (coder:) has been implemented")
    }
}


class SettingCell: BaseCell{
    
    var name: String?{
        didSet{
            nameLabel.text = name
        }
    }
    
    let nameLabel: UILabel = {
        let label = UILabel()
        label.text = "Portrait"
        return label
    }()
    
    override func setUpView() {
        super.setUpView()
        
        addSubview(nameLabel)
        
        addContraintsWithFormat("H:|-16-[v0]|", views: nameLabel)
        addContraintsWithFormat("V:|[v0]|", views: nameLabel)
        
    }
}




extension UIView{
    func addContraintsWithFormat(_ format: String, views: UIView...) {
        var viewDict = [String: UIView]()

        for (index, view) in views.enumerated() {
            let key = "v\(index)"
            view.translatesAutoresizingMaskIntoConstraints = false
            viewDict[key] = view
        }

        addConstraints(NSLayoutConstraint.constraints(withVisualFormat: format, options: NSLayoutConstraint.FormatOptions(rawValue: 0), metrics: nil, views: viewDict))
    }
}
